"""SRT and ASS subtitle export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theoria.config import TheoriaConfig


def format_timestamp(seconds: float) -> str:
    """Convert float seconds (19.24) to SRT format (00:00:19,240)."""
    total_ms = round(seconds * 1000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millisecs = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """Convert float seconds to ASS format (H:MM:SS.cc)."""
    total_cs = round(seconds * 100)
    hours, total_cs = divmod(total_cs, 360_000)
    minutes, total_cs = divmod(total_cs, 6_000)
    secs, centisecs = divmod(total_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def export_srt(translated_data: list[dict], output_path: str = "output/test_subs.srt", config: TheoriaConfig | None = None):
    """Write the JSON array out to a standard .srt file.

    Captions are written as separate SRT entries with their own timing,
    centered around the segment midpoint and split sequentially when
    multiple captions exist.
    """
    max_cap_duration = config.max_cap_duration if config else 5.0
    entries = []

    for line in translated_data:
        seg_start = line["start"]
        seg_end = line["end"]

        # Collect high-importance captions
        captions = []
        for cap in line.get("on_screen_captions", []):
            if isinstance(cap, dict) and cap.get("importance") == "high" and cap.get("text", "").strip():
                captions.append(cap["text"])

        # Write captions as separate entries with midpoint-centered timing
        if captions:
            mid = (seg_start + seg_end) / 2
            cap_window_start = max(seg_start, mid - max_cap_duration / 2)
            cap_window_end = min(seg_end, mid + max_cap_duration / 2)
            window_len = cap_window_end - cap_window_start

            slice_len = window_len / len(captions)
            for j, text in enumerate(captions):
                c_start = cap_window_start + j * slice_len
                c_end = cap_window_start + (j + 1) * slice_len
                entries.append((c_start, c_end, f"[{text}]"))

        # Write dialogue as its own entry
        if line.get("english_text", "").strip():
            entries.append((seg_start, seg_end, line["english_text"]))

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(entries, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")

    print(f"\nSRT successfully generated at {output_path}")


def export_ass(translated_data: list[dict], output_path: str = "output/test_subs.ass", config: TheoriaConfig | None = None):
    """Write the JSON array out to an Advanced Substation Alpha (.ass) file."""
    cfg_res_x = config.play_res_x if config else 1920
    cfg_res_y = config.play_res_y if config else 1080
    d_font = config.dialogue_font if config else "Arial"
    d_size = config.dialogue_fontsize if config else 60
    c_font = config.caption_font if config else "Arial"
    c_size = config.caption_fontsize if config else 45
    max_cap_duration = config.max_cap_duration if config else 5.0
    line_h = config.line_height if config else 55

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {cfg_res_x}",
        f"PlayResY: {cfg_res_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{d_font},{d_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,2,2,30,30,30,1",
        f"Style: Caption,{c_font},{c_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,30,30,30,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")

        # Coordinate mapping for PlayRes
        loc_map = {
            "top-left": (100, 100, 7), "top-center": (960, 100, 8), "top-right": (1820, 100, 9),
            "middle-left": (100, 540, 4), "middle-center": (960, 540, 5), "middle-right": (1820, 540, 6),
            "bottom-left": (100, 900, 1), "bottom-center": (960, 900, 2), "bottom-right": (1820, 900, 3),
        }

        for line in translated_data:
            start = format_timestamp_ass(line["start"])
            end = format_timestamp_ass(line["end"])

            if line.get("english_text", "").strip():
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line['english_text']}\n")

            if "on_screen_captions" in line and line["on_screen_captions"]:
                mid = (line["start"] + line["end"]) / 2
                cap_start = format_timestamp_ass(max(line["start"], mid - max_cap_duration / 2))
                cap_end = format_timestamp_ass(min(line["end"], mid + max_cap_duration / 2))

                zone_counts: dict = {}
                for caption in line["on_screen_captions"]:
                    text = caption if isinstance(caption, str) else caption.get("text", "")
                    loc_str = caption.get("location", "top-center") if isinstance(caption, dict) else "top-center"
                    x, y_base, align = loc_map.get(loc_str.lower(), (960, 100, 8))

                    count = zone_counts.get((x, y_base, align), 0)
                    zone_counts[(x, y_base, align)] = count + 1

                    if align in (1, 2, 3):
                        y = y_base - count * line_h
                    elif align in (7, 8, 9):
                        y = y_base + count * line_h
                    else:
                        y = y_base - count * line_h

                    f.write(f"Dialogue: 1,{cap_start},{cap_end},Caption,,0,0,0,,{{\\an{align}\\pos({x},{y})}}{text}\n")

    print(f"\nASS successfully generated at {output_path}")
