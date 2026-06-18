"""
Assembles highlight clips into a final music video.

Strategy:
  1. Rank all highlight segments across all clips by score.
  2. Assign the best segments to music peaks (drop/chorus moments).
  3. Fill remaining music beats with the next-best segments.
  4. Trim each segment to exactly fill its time slot (beat to next beat).
  5. Concatenate, overlay music, render.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
)

from src.ingestion.loader import Clip
from src.highlight.scorer import HighlightSegment
from src.audio.analyzer import MusicAnalysis


class _Assignment(NamedTuple):
    clip: Clip
    segment: HighlightSegment
    music_start: float   # where in the music track this clip starts
    music_end: float     # where it ends


def _collect_ranked_segments(clips: list[Clip]) -> list[tuple[Clip, HighlightSegment]]:
    all_segs = []
    for clip in clips:
        for seg in clip.highlight_segments:
            all_segs.append((clip, seg))
    all_segs.sort(key=lambda x: x[1].score, reverse=True)
    return all_segs


def plan_edit(
    clips: list[Clip],
    music: MusicAnalysis,
    max_duration: float | None = None,
) -> list[_Assignment]:
    """
    Match highlight segments to music beat slots.

    max_duration: cap the output video length (seconds). Defaults to music duration.
    """
    limit = min(music.duration, max_duration or music.duration)
    beat_times = music.beat_times[music.beat_times < limit]
    if len(beat_times) == 0:
        raise ValueError("No beats found within the requested duration.")

    ranked = _collect_ranked_segments(clips)
    if not ranked:
        raise ValueError("No highlight segments found. Lower the threshold or add more clips.")

    # Build beat slots: each slot is [beat_times[i], beat_times[i+1])
    slots: list[tuple[float, float]] = []
    for i in range(len(beat_times) - 1):
        slots.append((beat_times[i], beat_times[i + 1]))

    # Prioritize peak slots
    peak_set = set(music.peak_times[music.peak_times < limit].round(3))
    peak_slots = [s for s in slots if round(s[0], 3) in peak_set]
    normal_slots = [s for s in slots if s not in peak_slots]

    ordered_slots = peak_slots + normal_slots
    assignments: list[_Assignment] = []
    seg_iter = iter(ranked)

    for slot_start, slot_end in ordered_slots:
        try:
            clip, seg = next(seg_iter)
        except StopIteration:
            # Cycle ranked segments if we run out
            seg_iter = iter(ranked)
            clip, seg = next(seg_iter)

        assignments.append(_Assignment(
            clip=clip,
            segment=seg,
            music_start=slot_start,
            music_end=slot_end,
        ))

    return assignments


def render(
    assignments: list[_Assignment],
    music: MusicAnalysis,
    output_path: str | Path,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 60,
    gameplay_volume: float = 0.0,
    music_volume: float = 1.0,
) -> Path:
    """
    Render the final video from planned assignments.

    gameplay_volume: 0.0 = mute gameplay audio, 1.0 = full.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering {len(assignments)} clips → {output_path.name}")

    video_segments = []
    for a in assignments:
        slot_dur = a.music_end - a.music_start
        clip_dur = a.segment.end - a.segment.start

        vc = VideoFileClip(str(a.clip.path), audio=(gameplay_volume > 0))
        subclip = vc.subclip(
            min(a.segment.start, vc.duration - 0.1),
            min(a.segment.end, vc.duration),
        )

        # Speed-adjust if the clip is shorter than the slot (rare but possible)
        if subclip.duration < slot_dur:
            subclip = subclip.speedx(subclip.duration / slot_dur)
        else:
            subclip = subclip.subclip(0, slot_dur)

        subclip = subclip.resize(resolution)

        if gameplay_volume == 0.0:
            subclip = subclip.without_audio()
        elif gameplay_volume < 1.0 and subclip.audio is not None:
            subclip = subclip.set_audio(subclip.audio.volumex(gameplay_volume))

        video_segments.append(subclip)

    final_video = concatenate_videoclips(video_segments, method="compose")

    music_clip = AudioFileClip(str(music.path)).subclip(0, min(music.duration, final_video.duration))
    if music_volume != 1.0:
        music_clip = music_clip.volumex(music_volume)

    if gameplay_volume > 0.0 and final_video.audio is not None:
        mixed_audio = CompositeAudioClip([final_video.audio, music_clip])
        final_video = final_video.set_audio(mixed_audio)
    else:
        final_video = final_video.set_audio(music_clip)

    final_video.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        logger="bar",
    )

    print(f"\nDone. Output: {output_path}")
    return output_path
