"""CLI smoke tests."""

import subprocess
import sys


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
