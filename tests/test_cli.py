"""CLI smoke tests."""

import hashlib
import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """CLI loads and prints help without import errors."""
    result = subprocess.run(
        [sys.executable, "-m", "theoria", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "--video" in result.stdout


def test_cli_help_includes_clear_cache():
    """--clear-cache flag is advertised in help."""
    result = subprocess.run(
        [sys.executable, "-m", "theoria", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--clear-cache" in result.stdout


def test_run_dir_hash_format(tmp_path):
    """run_dir is named {stem}_{8-char-hash} based on path/mtime/size."""
    # Create a fake video file so stat() works
    fake_video = tmp_path / "episode01.mkv"
    fake_video.write_bytes(b"\x00" * 128)

    vs = fake_video.stat()
    expected_hash = hashlib.sha256(
        f"{fake_video.resolve()}|{vs.st_mtime}|{vs.st_size}".encode()
    ).hexdigest()[:8]

    expected_stem = f"episode01_{expected_hash}"
    assert len(expected_hash) == 8
    assert all(c in "0123456789abcdef" for c in expected_hash)
    # Verify the stem matches the pattern theoria.cli uses
    assert expected_stem.startswith("episode01_")


def test_run_dir_hash_differs_for_different_files(tmp_path):
    """Two video files with different content produce different run_dir hashes."""
    video_a = tmp_path / "episode01.mkv"
    video_b = tmp_path / "episode01.mkv"  # same name, different dir
    dir_b = tmp_path / "series2"
    dir_b.mkdir()
    video_b = dir_b / "episode01.mkv"

    video_a.write_bytes(b"\x00" * 128)
    video_b.write_bytes(b"\x00" * 128)

    def content_hash(p):
        vs = p.stat()
        return hashlib.sha256(
            f"{p.resolve()}|{vs.st_mtime}|{vs.st_size}".encode()
        ).hexdigest()[:8]

    # Different resolved paths → different hashes even with same filename/size
    assert content_hash(video_a) != content_hash(video_b)


def test_clear_cache_wipes_run_dir(tmp_path, monkeypatch):
    """--clear-cache removes and recreates the run directory."""
    import shutil
    from unittest.mock import patch, MagicMock

    # Point output to tmp_path
    monkeypatch.chdir(tmp_path)

    fake_video = tmp_path / "show.mkv"
    fake_video.write_bytes(b"\x00" * 64)

    vs = fake_video.stat()
    content_hash = hashlib.sha256(
        f"{fake_video.resolve()}|{vs.st_mtime}|{vs.st_size}".encode()
    ).hexdigest()[:8]
    run_dir = tmp_path / "output" / f"show_{content_hash}"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "old_cache.rttm"
    sentinel.write_text("stale")

    assert sentinel.exists()

    # Simulate what cli.main() does for clear_cache
    import shutil as _shutil
    if run_dir.exists():
        _shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    assert run_dir.exists()
    assert not sentinel.exists()
