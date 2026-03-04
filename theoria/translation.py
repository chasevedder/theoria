"""Gemini-based translation with response validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from google.genai import Client
    from theoria.config import TheoriaConfig


def validate_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Validate Gemini response segments, fixing or skipping malformed entries.

    Returns:
        Tuple of (valid_segments, skipped_count).
    """
    valid = []
    skipped = 0

    for seg in segments:
        # Must have numeric start and end
        if not (isinstance(seg.get("start"), (int, float)) and isinstance(seg.get("end"), (int, float))):
            print(f"  Warning: skipping segment with invalid start/end: {seg}")
            skipped += 1
            continue

        # Warn if english_text is missing but text exists
        if "english_text" not in seg and "text" in seg:
            print(f"  Warning: segment at {seg['start']:.2f}s missing english_text, falling back to original text")
            seg["english_text"] = seg.get("text", "")

        # Filter malformed on_screen_captions
        if "on_screen_captions" in seg:
            valid_caps = []
            for cap in seg["on_screen_captions"]:
                if isinstance(cap, dict) and "text" in cap:
                    valid_caps.append(cap)
            seg["on_screen_captions"] = valid_caps

        valid.append(seg)

    return valid, skipped


def translate_with_gemini(
    client: Client,
    video_file,
    aligned_data_chunk: list[dict],
    chunk_id: int = 0,
    context: str | None = None,
    run_dir=None,
    include_captions: bool = False,
    sample_rate: float = 0,
    config: TheoriaConfig | None = None,
):
    """Send frames + transcript to Gemini for translation.

    Returns:
        Tuple of (translated_segments, usage_metadata).
    """
    from google.genai import types

    # Pull config values with fallbacks
    model_id = config.gemini_model if config else "gemini-3-flash-preview"
    temperature = config.temperature if config else 1.0

    base_dir = Path(run_dir).resolve() if run_dir else Path("output").resolve()
    frames_dir = base_dir / f"frames_{chunk_id}"

    # Build rules without inline numbers so insertion order always produces correct numbering
    rules = [
        "Return the EXACT same JSON structure, but look at the corresponding video frame to identify who is speaking. Overwrite the generic 'speaker' label with their actual name or a visual description (e.g., 'Cast Member in Red').",
        "Add an 'english_text' key with the translation, adapting slang/idioms.",
    ]

    if include_captions:
        rules.append(
            "Look for on-screen text/captions in the frames. If found, add an 'on_screen_captions' array to that segment. "
            "Each item in the array should be an object with 'text' (English translation), 'location' (one of: 'top-left', "
            "'top-center', 'top-right', 'middle-left', 'middle-center', 'middle-right', 'bottom-left', 'bottom-center', "
            "'bottom-right'), and 'importance' ('high' or 'low')."
        )
        rules.append(
            "Mark importance as 'high' for captions that provide meaningful information a viewer needs: names/titles, "
            "locations, timestamps, explanations, context, or narrative commentary. Mark as 'low' for captions that are "
            "expressions, reactions, sound effects, emphasis of spoken words, or anything that doesn't add substantive "
            "value beyond the dialogue."
        )
        genre_rules = config.get_genre_rules() if config else ""
        if genre_rules:
            rules.append(genre_rules)

    rules.append("Output strictly valid JSON.")
    numbered_rules = [f"{i+1}. {r}" for i, r in enumerate(rules)]

    intro = config.get_prompt_intro() if config else "You are an expert Korean to English translator specializing in variety shows."
    prompt_contents = [
        f"{intro}\n"
        "You will receive a sequence of video frames corresponding to spoken lines, followed by a JSON array of the transcript.\n\n"
        "Rules:\n" + "\n".join(numbered_rules)
    ]

    if context:
        prompt_contents.append(f"Context from previous segments (use for name/term consistency):\n{context}")

    if sample_rate > 0:
        earliest_start = min(s["start"] for s in aligned_data_chunk)
        latest_end = max(s["end"] for s in aligned_data_chunk)
        duration = latest_end - earliest_start
        num_samples = int(duration / sample_rate) + 1

        for i in range(num_samples):
            frame_path = frames_dir / f"sample_{i}.jpg"
            if frame_path.exists():
                timestamp = earliest_start + (i * sample_rate)
                prompt_contents.append(f"Video Frame at {timestamp:.2f}s:")
                prompt_contents.append(Image.open(frame_path))
    else:
        for i, segment in enumerate(aligned_data_chunk):
            frame_path = frames_dir / f"chunk_{i}.jpg"
            prompt_contents.append(f"Frame for segment {i} ({segment['start']:.2f}s -> {segment['end']:.2f}s):")
            if frame_path.exists():
                prompt_contents.append(Image.open(frame_path))

    transcript_json = json.dumps(aligned_data_chunk, ensure_ascii=False, indent=2)
    prompt_contents.append(f"Transcript JSON:\n{transcript_json}")

    response = client.models.generate_content(
        model=model_id,
        contents=prompt_contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )

    usage = response.usage_metadata

    # Handle empty response
    if not response.text or not response.text.strip():
        print(f"  [Chunk {chunk_id}] Gemini returned empty response.")
        return None, usage

    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed = json.loads(raw_text.strip())
        validated, skipped = validate_segments(parsed)
        if skipped > 0:
            print(f"  [Chunk {chunk_id}] Validated: {skipped} segment(s) skipped.")
        return validated, usage
    except json.JSONDecodeError:
        print("Gemini failed to return valid JSON.")
        print(f"Raw response: {response.text}")
        return None, usage
