"""Transition effects between gameplay clips."""
from __future__ import annotations

from enum import Enum

import numpy as np
from moviepy.editor import VideoClip, concatenate_videoclips, ColorClip, CompositeVideoClip


class TransitionType(str, Enum):
    HARD_CUT = "cut"
    FLASH_WHITE = "flash-white"
    FLASH_BLACK = "flash-black"
    WHIP_ZOOM = "whip-zoom"


def apply_transition(
    clip_a: VideoClip,
    clip_b: VideoClip,
    transition: TransitionType,
    duration: float = 0.08,
) -> VideoClip:
    """
    Return a single VideoClip that is clip_a → transition → clip_b.
    duration: length of the transition effect in seconds.
    """
    if transition == TransitionType.HARD_CUT:
        return concatenate_videoclips([clip_a, clip_b], method="compose")

    if transition in (TransitionType.FLASH_WHITE, TransitionType.FLASH_BLACK):
        color = [255, 255, 255] if transition == TransitionType.FLASH_WHITE else [0, 0, 0]
        return _flash_transition(clip_a, clip_b, color, duration)

    if transition == TransitionType.WHIP_ZOOM:
        return _whip_zoom_transition(clip_a, clip_b, duration)

    return concatenate_videoclips([clip_a, clip_b], method="compose")


def _flash_transition(
    clip_a: VideoClip,
    clip_b: VideoClip,
    color: list[int],
    duration: float,
) -> VideoClip:
    """Flash to a solid color between clips."""
    w, h = clip_a.size
    half = duration / 2.0

    # clip_a fades to color over `half` seconds
    def make_fade_out(clip: VideoClip, to_color: list[int], fade_dur: float) -> VideoClip:
        def filter_frame(get_frame, t):
            frame = get_frame(t)
            alpha = t / fade_dur  # 0 at start, 1 at end
            overlay = np.array(to_color, dtype=np.float32)
            return np.clip(frame * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)
        trimmed = clip.subclip(max(0, clip.duration - fade_dur))
        return trimmed.fl(filter_frame, apply_to=["mask", "video"])

    def make_fade_in(clip: VideoClip, from_color: list[int], fade_dur: float) -> VideoClip:
        def filter_frame(get_frame, t):
            frame = get_frame(t)
            alpha = t / fade_dur  # 0 at start, 1 at end
            overlay = np.array(from_color, dtype=np.float32)
            return np.clip(frame * alpha + overlay * (1 - alpha), 0, 255).astype(np.uint8)
        trimmed = clip.subclip(0, min(clip.duration, fade_dur))
        return trimmed.fl(filter_frame, apply_to=["mask", "video"])

    a_body = clip_a.subclip(0, max(0, clip_a.duration - half))
    a_fade = make_fade_out(clip_a, color, half)
    b_fade = make_fade_in(clip_b, color, half)
    b_body = clip_b.subclip(min(clip_b.duration, half))

    return concatenate_videoclips([a_body, a_fade, b_fade, b_body], method="compose")


def _whip_zoom_transition(
    clip_a: VideoClip,
    clip_b: VideoClip,
    duration: float,
) -> VideoClip:
    """Quick zoom-in on the last frame of clip_a, zoom-out on the first of clip_b."""
    half = duration / 2.0

    def zoom_in_filter(get_frame, t):
        frame = get_frame(t)
        scale = 1.0 + (t / half) * 0.3  # 1.0 → 1.3
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        return cv2.warpAffine(frame, M, (w, h))

    def zoom_out_filter(get_frame, t):
        frame = get_frame(t)
        scale = 1.3 - (t / half) * 0.3  # 1.3 → 1.0
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        return cv2.warpAffine(frame, M, (w, h))

    import cv2 as _cv2
    # Reuse cv2 in closures
    def zoom_in_filter(get_frame, t):  # noqa: F811
        frame = get_frame(t)
        scale = 1.0 + (t / half) * 0.3
        h, w = frame.shape[:2]
        M = _cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        return _cv2.warpAffine(frame, M, (w, h))

    def zoom_out_filter(get_frame, t):  # noqa: F811
        frame = get_frame(t)
        scale = 1.3 - (t / half) * 0.3
        h, w = frame.shape[:2]
        M = _cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        return _cv2.warpAffine(frame, M, (w, h))

    a_body = clip_a.subclip(0, max(0, clip_a.duration - half))
    a_zoom = clip_a.subclip(max(0, clip_a.duration - half)).fl(zoom_in_filter)
    b_zoom = clip_b.subclip(0, min(clip_b.duration, half)).fl(zoom_out_filter)
    b_body = clip_b.subclip(min(clip_b.duration, half))

    return concatenate_videoclips([a_body, a_zoom, b_zoom, b_body], method="compose")
