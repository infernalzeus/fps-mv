"""Per-clip visual effects: color grading and speed ramp."""
from __future__ import annotations

import numpy as np
from moviepy.editor import VideoClip, concatenate_videoclips


def color_grade(
    clip: VideoClip,
    saturation: float = 1.4,
    contrast: float = 1.1,
    brightness: float = 0.0,
) -> VideoClip:
    """
    Boost saturation and contrast on a clip.
    saturation: multiplier on HSV S channel (>1 = more vivid)
    contrast: linear multiplier on brightness (>1 = more contrast)
    brightness: additive offset (-128 to 128)
    """
    import cv2

    def process(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * contrast + brightness, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return clip.fl_image(process)


def speed_ramp(
    clip: VideoClip,
    kill_time: float,
    slow_factor: float = 0.4,
    slow_window: float = 1.5,
) -> VideoClip:
    """
    Slow the clip around the kill moment, play the rest at normal speed.

    kill_time: seconds from the clip's own start where the kill happens
    slow_factor: playback speed during the slow window (0.4 = 40% speed)
    slow_window: how many seconds to slow down around the kill
    """
    clip_dur = clip.duration
    slow_start = max(0.0, kill_time - slow_window / 2)
    slow_end = min(clip_dur, kill_time + slow_window / 2)

    parts = []
    if slow_start > 0.01:
        parts.append(clip.subclip(0, slow_start))
    if slow_end > slow_start:
        parts.append(clip.subclip(slow_start, slow_end).speedx(slow_factor))
    if slow_end < clip_dur - 0.01:
        parts.append(clip.subclip(slow_end, clip_dur))

    if not parts:
        return clip

    return concatenate_videoclips(parts, method="compose")
