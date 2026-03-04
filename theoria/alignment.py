"""Whisper + Pyannote alignment into speaker-attributed segments."""

import bisect


def align_data(whisper_generator, diarization, max_duration=4.0, min_gap=0.5):
    """Align Whisper word timestamps with Pyannote speaker diarization.

    Args:
        whisper_generator: Whisper segments with word-level timestamps.
        diarization: Pyannote diarization Annotation object.
        max_duration: Maximum chunk duration in seconds before splitting.
        min_gap: Minimum silence gap (seconds) to fill with an empty segment.
    """
    aligned_transcript = []

    # Pre-sort diarization turns once for O(log n) per-word speaker lookup
    diar_turns = sorted(
        [(turn.start, turn.end, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)],
        key=lambda x: x[0],
    )
    diar_starts = [t[0] for t in diar_turns]

    def find_speaker(midpoint):
        idx = bisect.bisect_right(diar_starts, midpoint) - 1
        if idx >= 0 and diar_turns[idx][1] >= midpoint:
            return diar_turns[idx][2]
        return "UNKNOWN"

    for segment in whisper_generator:
        current_chunk_words = []
        chunk_start = 0.0

        for i, word in enumerate(segment.words):
            if not current_chunk_words:
                chunk_start = word.start

            current_chunk_words.append(word.word)
            current_duration = word.end - chunk_start

            if current_duration >= max_duration or i == len(segment.words) - 1:
                midpoint = chunk_start + (word.end - chunk_start) / 2
                current_speaker = find_speaker(midpoint)

                aligned_transcript.append({
                    "start": chunk_start,
                    "end": word.end,
                    "speaker": current_speaker,
                    "text": "".join(current_chunk_words).strip(),
                })

                current_chunk_words = []

    # Fill gaps between segments to capture on-screen text during silence
    filled_transcript = []
    for i, entry in enumerate(aligned_transcript):
        if i > 0:
            prev_end = filled_transcript[-1]["end"]
            curr_start = entry["start"]
            if curr_start - prev_end > min_gap:
                filled_transcript.append({
                    "start": prev_end,
                    "end": curr_start,
                    "speaker": "N/A",
                    "text": "",
                })
        else:
            if entry["start"] > min_gap:
                filled_transcript.append({
                    "start": 0.0,
                    "end": entry["start"],
                    "speaker": "N/A",
                    "text": "",
                })
        filled_transcript.append(entry)

    return filled_transcript
