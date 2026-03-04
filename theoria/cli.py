"""Command-line interface for theoria."""

import argparse
import sys
from pathlib import Path

from theoria import suppress_warnings
from theoria.config import PRESETS, load_config


def main():
    suppress_warnings()

    parser = argparse.ArgumentParser(description="Multimodal AI Subtitle Generator")

    # Required
    parser.add_argument("-v", "--video", required=True, type=str, help="Path to the source video file")

    # Optional
    parser.add_argument("-o", "--output", type=str, help="Path for the output subtitle file")
    parser.add_argument("--lang", type=str, default="ko", help="Source language code (default: 'ko')")
    parser.add_argument("--batch-size", type=int, help="Number of segments per Gemini API call")
    parser.add_argument("--max-workers", type=int, help="Level of parallelization for API calls")
    parser.add_argument("--sequential", action="store_true", help="Process batches sequentially to maintain context")
    parser.add_argument("--format", nargs="+", choices=["srt", "ass"], default=["srt"], help="Output format(s): srt, ass, or both (default: srt)")
    parser.add_argument("--sample-rate", type=float, default=0, help="Fixed interval for frame sampling in seconds (default: 0, uses dialogue midpoints)")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not delete temporary files on success")
    parser.add_argument("--gemini-model", type=str, help="Gemini model ID to use for translation")
    parser.add_argument("--detect-scenes", action="store_true", help="Attempt to pick the best frame via scene detection")
    parser.add_argument("--sample-duration", type=int, help="Limit audio extraction to N seconds for testing")
    parser.add_argument("--limit-segments", type=int, help="Limit processing to the first N dialogue segments")

    # Config
    parser.add_argument("--config", type=str, help="Path to theoria.toml config file")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Translation preset (default: variety). Use custom_prompt in theoria.toml for full control.")

    args = parser.parse_args()

    # Load config: defaults -> TOML -> CLI overrides
    config = load_config(args.config)

    # CLI overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_workers is not None:
        config.max_workers = args.max_workers
    if args.gemini_model is not None:
        config.gemini_model = args.gemini_model
    if args.preset is not None:
        config.preset = args.preset

    # Validate video file
    video_file = Path(args.video)
    if not video_file.exists():
        print(f"ERROR: Video file '{video_file}' not found.")
        sys.exit(1)

    # Setup output paths
    output_base = Path("output")
    output_base.mkdir(exist_ok=True)

    run_dir = output_base / video_file.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    audio_file = run_dir / f"{video_file.stem}_audio.wav"

    formats = list(dict.fromkeys(args.format))
    if len(formats) == 1 and args.output:
        output_paths = {formats[0]: Path(args.output)}
    else:
        if args.output and len(formats) > 1:
            print("Note: --output ignored when multiple formats are specified; paths are auto-derived.")
        base = Path(video_file.stem)
        output_paths = {fmt: base.with_suffix(f".{fmt}") for fmt in formats}

    print("--- Pipeline Initialization ---")
    print(f"Source Video: {video_file.name}")
    print(f"Target Audio: {audio_file}")
    for fmt, path in output_paths.items():
        print(f"Target Subtitles ({fmt.upper()}): {path}")
    print()

    # Extract audio
    from theoria.audio import extract_audio
    extract_audio(str(video_file), str(audio_file), duration=args.sample_duration)

    # Run pipeline
    from theoria.pipeline import run_pipeline
    run_pipeline(
        str(video_file),
        str(audio_file),
        {fmt: str(path) for fmt, path in output_paths.items()},
        lang=args.lang,
        sequential=args.sequential,
        detect_scenes=args.detect_scenes,
        run_dir=run_dir,
        limit_segments=args.limit_segments,
        sample_rate=args.sample_rate,
        no_cleanup=args.no_cleanup,
        config=config,
    )
