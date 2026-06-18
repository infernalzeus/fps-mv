"""Loads and validates video clip files."""
import os
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class Clip:
    path: Path
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    # Populated by highlight detector
    highlight_segments: list = field(default_factory=list)


def load_clips(paths: list[str]) -> list[Clip]:
    """Validate and probe each clip path, returning Clip objects."""
    import cv2

    clips = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Clip not found: {p}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {path.suffix}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {p}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = frame_count / fps if fps > 0 else 0.0
        clips.append(Clip(path=path, duration=duration, fps=fps, width=width, height=height))

    return clips
