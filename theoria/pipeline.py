"""Main processing pipeline."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from theoria.alignment import align_data
from theoria.exporters import export_ass, export_srt
from theoria.translation import translate_with_gemini

if TYPE_CHECKING:
    from theoria.config import TheoriaConfig


def run_pipeline(
    video_file: str,
    audio_file: str,
    output_paths: dict[str, str],
    lang: str = "ko",
    sequential: bool = False,
    detect_scenes: bool = False,
    run_dir=None,
    limit_segments: int | None = None,
    sample_rate: float = 0,
    no_cleanup: bool = False,
    config: TheoriaConfig | None = None,
):
    # Import config defaults if not provided
    if config is None:
        from theoria.config import TheoriaConfig
        config = TheoriaConfig()

    batch_size = config.batch_size
    max_workers = config.max_workers

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    warnings.filterwarnings("ignore", message=".*weights_only=False.*")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return

    from google import genai
    gemini_client = genai.Client(api_key=api_key)

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n")

    # 1. Load Whisper
    print(f"Loading Whisper {config.whisper_model}...")
    whisper_model = WhisperModel(
        config.whisper_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
    )

    # 2. Load Pyannote
    print("Loading Pyannote Diarization...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        return

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
        diarization_pipeline.batch_size = 32

    print("\n--- Starting Processing ---")
    start_time = time.time()

    overall_progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        auto_refresh=False,
    )
    step_progress = Progress(
        TextColumn("  ┗━ [bold i]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        auto_refresh=False,
    )
    progress_group = Group(
        Panel(overall_progress, title="Overall Status", border_style="blue"),
        step_progress,
    )

    with Live(progress_group, refresh_per_second=4):
        overall_task = overall_progress.add_task("Total Pipeline", total=5)
        base_run_dir = run_dir if run_dir else Path("output")

        def run_with_refresh(func, *args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                while True:
                    done, _ = wait([future], timeout=0.25)
                    if done:
                        return future.result()

        # Step A: Run Diarization
        diar_task = step_progress.add_task("Speaker Diarization", total=None)
        rttm_path = base_run_dir / "diarization.rttm"

        if rttm_path.exists():
            try:
                from pyannote.database.util import load_rttm
                diarization = list(load_rttm(rttm_path).values())[0]
                step_progress.update(diar_task, completed=1, total=1, description="Diarization Loaded from Cache")
            except Exception as e:
                print(f"  Warning: corrupt diarization cache ({e}), re-running...")
                rttm_path.unlink(missing_ok=True)
                diarization = run_with_refresh(diarization_pipeline, audio_file)
                with open(rttm_path, "w") as f:
                    diarization.write_rttm(f)
                step_progress.update(diar_task, completed=1, total=1, description="Diarization Complete")
        else:
            diarization = run_with_refresh(diarization_pipeline, audio_file)
            with open(rttm_path, "w") as f:
                diarization.write_rttm(f)
            step_progress.update(diar_task, completed=1, total=1, description="Diarization Complete")
        overall_progress.update(overall_task, advance=1)

        # Step B: Run Whisper
        whisper_task = step_progress.add_task("Whisper Transcription", total=None)
        whisper_cache_path = base_run_dir / "whisper_segments.pkl"

        if whisper_cache_path.exists():
            try:
                with open(whisper_cache_path, "rb") as f:
                    segments = pickle.load(f)
                step_progress.update(whisper_task, completed=1, total=1, description="Transcription Loaded from Cache")
            except Exception as e:
                print(f"  Warning: corrupt whisper cache ({e}), re-running...")
                whisper_cache_path.unlink(missing_ok=True)
                def transcribe_full(*args, **kwargs):
                    segs, info = whisper_model.transcribe(*args, **kwargs)
                    return list(segs), info
                segments, info = run_with_refresh(
                    transcribe_full, audio_file, beam_size=1, language=lang,
                    condition_on_previous_text=False, vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500), word_timestamps=True,
                )
                with open(whisper_cache_path, "wb") as f:
                    pickle.dump(segments, f)
                step_progress.update(whisper_task, completed=1, total=1, description="Transcription Complete")
        else:
            def transcribe_full(*args, **kwargs):
                segs, info = whisper_model.transcribe(*args, **kwargs)
                return list(segs), info
            segments, info = run_with_refresh(
                transcribe_full, audio_file, beam_size=1, language=lang,
                condition_on_previous_text=False, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500), word_timestamps=True,
            )
            with open(whisper_cache_path, "wb") as f:
                pickle.dump(segments, f)
            step_progress.update(whisper_task, completed=1, total=1, description="Transcription Complete")
        overall_progress.update(overall_task, advance=1)

        # Step C: Align the Data
        align_task = step_progress.add_task("Aligning Timestamps", total=None)
        segments_json_path = base_run_dir / "segments.json"

        if segments_json_path.exists():
            with open(segments_json_path, "r", encoding="utf-8") as f:
                final_data = json.load(f)
            step_progress.update(align_task, completed=1, total=1, description="Segments Loaded from Cache")
        else:
            final_data = run_with_refresh(
                align_data, segments, diarization,
                max_duration=config.max_duration, min_gap=config.min_gap,
            )
            with open(segments_json_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            step_progress.update(align_task, completed=1, total=1, description="Alignment Complete")

        if limit_segments:
            final_data = final_data[:limit_segments]
        overall_progress.update(overall_task, advance=1)

        # Step D: Extract Video Frames
        total_batches = (len(final_data) + batch_size - 1) // batch_size
        chunks = [final_data[i : i + batch_size] for i in range(0, len(final_data), batch_size)]
        extract_task = step_progress.add_task("Extracting Frames", total=total_batches)

        def extract_chunk_frames(idx, chunk, sample_rate=0):
            frames_dir = Path(base_run_dir).resolve() / f"frames_{idx}"

            earliest_start = min(segment["start"] for segment in chunk)
            latest_end = max(segment["end"] for segment in chunk)
            seek_start = max(0.0, earliest_start - 1)
            seek_offset = earliest_start - seek_start

            if sample_rate > 0:
                duration = latest_end - earliest_start
                expected_count = int(duration / sample_rate) + 1
            else:
                expected_count = len(chunk)

            if frames_dir.exists():
                existing_frames = [f for f in frames_dir.glob("*.jpg") if f.stat().st_size > 0]
                if len(existing_frames) == expected_count:
                    return

            frames_dir.mkdir(parents=True, exist_ok=True)
            filter_parts = []
            map_args = []

            if sample_rate > 0:
                for i in range(expected_count):
                    t = seek_offset + i * sample_rate
                    out_path = str(frames_dir / f"sample_{i}.jpg")
                    filter_parts.append(f"[0:v]select='gte(t,{t})',setpts=N/FRAME_RATE/TB[v{i}]")
                    map_args.append(["-map", f"[v{i}]", "-vframes", "1", "-pix_fmt", "yuvj420p", "-update", "1", out_path])
            else:
                for i, segment in enumerate(chunk):
                    out_path = str(frames_dir / f"chunk_{i}.jpg")
                    if detect_scenes:
                        rel_start = max(0.0, segment["start"] - seek_start)
                        rel_end = segment["end"] - seek_start
                        select_expr = "gt(scene,0.1)+eq(n,0)"
                        filter_parts.append(f"[0:v]trim=start={rel_start}:end={rel_end},scdet=threshold=10,select='{select_expr}',setpts=N/FRAME_RATE/TB[v{i}]")
                    else:
                        rel_midpoint = max(0.0, (segment["start"] + segment["end"]) / 2 - seek_start)
                        filter_parts.append(f"[0:v]select='gte(t,{rel_midpoint})',setpts=N/FRAME_RATE/TB[v{i}]")
                    map_args.append(["-map", f"[v{i}]", "-vframes", "1", "-pix_fmt", "yuvj420p", "-update", "1", out_path])

            filter_expr = ";".join(filter_parts)
            scan_duration = (latest_end - earliest_start) + 2

            ffmpeg_cmd = [
                "ffmpeg", "-hwaccel", "auto",
                "-ss", str(seek_start), "-t", str(scan_duration),
                "-hide_banner", "-loglevel", "error", "-y", "-strict", "-2",
                "-i", str(video_file),
                "-filter_complex", filter_expr,
            ]
            for args in map_args:
                ffmpeg_cmd.extend(args)

            try:
                subprocess.run(ffmpeg_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  Warning: frame extraction failed for batch {idx}: {e}")

        extraction_workers = min(max_workers, 6)
        with ThreadPoolExecutor(max_workers=extraction_workers) as executor:
            futures = [executor.submit(extract_chunk_frames, i, chunk, sample_rate) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Warning: frame extraction error: {e}")
                step_progress.update(extract_task, advance=1)
        step_progress.update(extract_task, description="Frame Extraction Complete")
        overall_progress.update(overall_task, advance=1)

        # Step E: Translate with Gemini
        translate_task = step_progress.add_task("Gemini Translation", total=total_batches)
        results_map = {}
        failed_chunks = []
        total_input_tokens = 0
        total_output_tokens = 0

        def process_chunk(idx, chunk, context=None):
            include_captions = "ass" in output_paths or "srt" in output_paths
            for attempt in range(config.max_retries):
                try:
                    translated_chunk, usage = translate_with_gemini(
                        gemini_client, video_file, chunk, chunk_id=idx,
                        context=context, run_dir=base_run_dir,
                        include_captions=include_captions, sample_rate=sample_rate,
                        config=config,
                    )
                    if translated_chunk:
                        return idx, translated_chunk, usage
                except Exception as e:
                    print(f"  [Chunk {idx}] Attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_sleep)
            return idx, None, None

        if sequential:
            current_context = ""
            for i, chunk in enumerate(chunks):
                idx, res, usage = process_chunk(i, chunk, context=current_context)
                if res:
                    results_map[idx] = res
                    if usage:
                        total_input_tokens += usage.prompt_token_count or 0
                        total_output_tokens += usage.candidates_token_count or 0
                    summary_data = [{"speaker": d.get("speaker"), "english_text": d.get("english_text")} for d in res[-config.context_window_size:]]
                    current_context = json.dumps(summary_data, ensure_ascii=False)
                else:
                    failed_chunks.append(idx)
                step_progress.update(translate_task, advance=1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}
                while futures:
                    done, not_done = wait(futures.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, res, usage = future.result()
                        if res:
                            results_map[idx] = res
                            if usage:
                                total_input_tokens += usage.prompt_token_count or 0
                                total_output_tokens += usage.candidates_token_count or 0
                        else:
                            failed_chunks.append(idx)
                        step_progress.update(translate_task, advance=1)
                        del futures[future]
        step_progress.update(translate_task, description="Translation Complete")
        overall_progress.update(overall_task, advance=1)

    if failed_chunks:
        failed_str = ", ".join(str(i) for i in sorted(failed_chunks))
        print(f"\nWarning: chunks {failed_str} failed translation — output has gaps")

    all_translated_data = []
    for i in range(len(chunks)):
        if i in results_map:
            all_translated_data.extend(results_map[i])

    translated_json_path = base_run_dir / "translated_results.json"
    with open(translated_json_path, "w", encoding="utf-8") as f:
        json.dump(all_translated_data, f, ensure_ascii=False, indent=2)

    if all_translated_data:
        if "srt" in output_paths:
            def has_srt_content(d):
                if d.get("english_text", "").strip():
                    return True
                for cap in d.get("on_screen_captions", []):
                    if isinstance(cap, dict) and cap.get("importance") == "high" and cap.get("text", "").strip():
                        return True
                return False
            srt_data = [d for d in all_translated_data if has_srt_content(d)]
            export_srt(srt_data, output_paths["srt"], config=config)

        if "ass" in output_paths:
            export_ass(all_translated_data, output_paths["ass"], config=config)

        all_chunks_succeeded = len(results_map) == len(chunks)
        if not no_cleanup and base_run_dir.exists() and all_chunks_succeeded:
            print(f"Cleaning up temporary files in {base_run_dir}...")
            shutil.rmtree(base_run_dir)
    else:
        print("Translation failed. No output generated.")

    print(f"\nTotal tokens: {total_input_tokens} input, {total_output_tokens} output")

    from theoria.config import DEFAULT_GEMINI_MODEL
    using_default_model = config.gemini_model == DEFAULT_GEMINI_MODEL
    if using_default_model or config._costs_from_config:
        input_cost = (total_input_tokens / 1_000_000) * config.input_cost_per_million
        output_cost = (total_output_tokens / 1_000_000) * config.output_cost_per_million
        total_cost = input_cost + output_cost
        pricing_source = DEFAULT_GEMINI_MODEL if using_default_model else config.gemini_model
        print(f"Estimated cost: ${total_cost:.4f} (based on {pricing_source} pricing)")
    else:
        print(
            f"Cost estimate unavailable — model '{config.gemini_model}' differs from default. "
            "Set input_cost_per_million/output_cost_per_million in theoria.toml to enable."
        )

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
