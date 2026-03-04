"""Audio extraction from video files."""

import subprocess
import sys
from pathlib import Path


def extract_audio(video_path: str, audio_path: str, duration: int = None):
    """Extract 16kHz mono audio specifically formatted for Whisper/Pyannote."""
    if Path(audio_path).exists():
        print(f"Found existing audio track at {audio_path}. Skipping extraction...")
        return

    print(f"Extracting 16kHz mono audio to {audio_path}...")
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if duration:
            cmd += ["-t", str(duration)]

        cmd += [
            "-i", video_path,
            "-map", "0:a:0",
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path,
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\nERROR: FFmpeg failed to extract audio. Is the video path correct?")
        sys.exit(1)
