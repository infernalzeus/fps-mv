"""Verify ffmpeg is installed and accessible."""
import subprocess
import sys


def require_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found. Install it and make sure it's on your PATH.")
        print("  Windows: winget install ffmpeg  OR  choco install ffmpeg")
        sys.exit(1)
