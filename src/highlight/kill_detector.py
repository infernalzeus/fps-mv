"""
Kill event detection via kill-feed region analysis.

Strategy: crop each sampled frame to the kill-feed ROI, compute mean brightness
of the top strip. A new kill entry flashes brighter than the steady-state.
We flag seconds where brightness delta exceeds a threshold as kill events.

No ML/OCR needed — the flash from a new kill entry appearing is reliable
enough for FPS games. Falls back to empty list if no ROI defined.
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from src.highlight.game_profiles import GameProfile


def detect_kill_events(
    video_path: Path,
    profile: GameProfile,
    sample_fps: float = 8.0,
    brightness_delta_threshold: float = 15.0,
) -> list[float]:
    """
    Return a list of timestamps (seconds) where a kill event was detected.

    brightness_delta_threshold: mean brightness change in the ROI that
    constitutes a new kill-feed entry (0–255 scale).
    """
    if profile.kill_feed_roi is None:
        return []

    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        cap.release()
        return []

    rx, ry, rw, rh = profile.kill_feed_roi
    step = max(1, int(native_fps / sample_fps))
    prev_brightness: float | None = None
    kill_times: list[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            x0 = int(rx * w)
            y0 = int(ry * h)
            x1 = int((rx + rw) * w)
            y1 = int((ry + rh) * h)
            roi = frame[y0:y1, x0:x1]

            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())

                if prev_brightness is not None:
                    delta = abs(brightness - prev_brightness)
                    if delta > brightness_delta_threshold:
                        t = frame_idx / native_fps
                        kill_times.append(t)

                prev_brightness = brightness

        frame_idx += 1

    cap.release()

    # Deduplicate: keep only one event per second
    seen: set[int] = set()
    deduped: list[float] = []
    for t in kill_times:
        sec = int(t)
        if sec not in seen:
            seen.add(sec)
            deduped.append(t)

    return deduped
