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
    VideoClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeAudioClip,
)

from src.ingestion.loader import Clip
from src.highlight.scorer import HighlightSegment
from src.audio.analyzer import MusicAnalysis
from src.editor.transitions import TransitionType, apply_transition
from src.editor.effects import color_grade, speed_ramp


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

    Peak slots (drop/chorus) get the highest-scoring segments.
    Slots remain in temporal (music) order so the output video follows the song.
    max_duration: cap the output video length (seconds). Defaults to music duration.
    """
    limit = min(music.duration, max_duration or music.duration)
    beat_times = music.beat_times[music.beat_times < limit]
    if len(beat_times) == 0:
        raise ValueError("No beats found within the requested duration.")

    ranked = _collect_ranked_segments(clips)
    if not ranked:
        raise ValueError("No highlight segments found. Lower the threshold or add more clips.")

    # Build slots in temporal order: [beat_times[i], beat_times[i+1])
    slots: list[tuple[float, float]] = [
        (beat_times[i], beat_times[i + 1]) for i in range(len(beat_times) - 1)
    ]

    # Mark which slot indices are peaks
    peak_set = set(music.peak_times[music.peak_times < limit].round(3))
    peak_slot_indices = {i for i, s in enumerate(slots) if round(s[0], 3) in peak_set}
    n_peaks = len(peak_slot_indices)

    # Reserve the top n_peaks ranked segments for peak slots.
    # This ensures the best clips land on the loudest moments.
    reserved_indices = set(range(min(n_peaks, len(ranked))))
    available = list(enumerate(ranked))  # [(pool_idx, (clip, seg)), ...]

    used_pairs: set[tuple[str, float]] = set()  # (clip_path, seg_start)
    last_clip_at: dict[str, int] = {}           # clip_path → last temporal slot_idx used

    def _pick(slot_idx: int, allow_reserved: bool) -> tuple[Clip, HighlightSegment]:
        """Pick best available segment respecting diversity constraints."""
        # Pass 1: full constraints (pair uniqueness + consecutive gap + reserved restriction)
        for pool_idx, (clip, seg) in available:
            if not allow_reserved and pool_idx in reserved_indices:
                continue
            if (str(clip.path), seg.start) in used_pairs:
                continue
            if last_clip_at.get(str(clip.path), -99) >= slot_idx - 1:
                continue
            used_pairs.add((str(clip.path), seg.start))
            last_clip_at[str(clip.path)] = slot_idx
            return clip, seg
        # Pass 2: relax consecutive constraint (keep pair uniqueness)
        for pool_idx, (clip, seg) in available:
            if (str(clip.path), seg.start) in used_pairs:
                continue
            used_pairs.add((str(clip.path), seg.start))
            last_clip_at[str(clip.path)] = slot_idx
            return clip, seg
        # Pass 3: all unique segments exhausted — cycle from best score
        best_clip, best_seg = available[0][1]
        last_clip_at[str(best_clip.path)] = slot_idx
        return best_clip, best_seg

    # Assign in temporal order; peak slots can dip into the reserved pool
    assignments: list[_Assignment] = []
    for slot_idx, (slot_start, slot_end) in enumerate(slots):
        is_peak = slot_idx in peak_slot_indices
        clip, seg = _pick(slot_idx, allow_reserved=is_peak)
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
    transition: TransitionType = TransitionType.FLASH_WHITE,
    apply_color_grade: bool = True,
    apply_speed_ramp: bool = True,
    preview: bool = False,
) -> Path:
    """
    Render the final video from planned assignments.

    gameplay_volume : 0.0 = mute gameplay audio, 1.0 = full.
    transition      : cut style between clips.
    apply_color_grade: boost saturation/contrast on each clip.
    apply_speed_ramp : slow-mo at kill moment when kill_time is known.
    preview         : fast low-res draft (480p, 24fps, hard cuts, no effects).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if preview:
        resolution = (854, 480)
        fps = 24
        transition = TransitionType.HARD_CUT
        apply_color_grade = False
        apply_speed_ramp = False

    print(f"\nRendering {len(assignments)} clips → {output_path.name}"
          + (" [PREVIEW]" if preview else ""))

    video_segments: list[VideoFileClip] = []
    for a in assignments:
        slot_dur = a.music_end - a.music_start

        vc = VideoFileClip(str(a.clip.path), audio=(gameplay_volume > 0))
        subclip = vc.subclip(
            min(a.segment.start, vc.duration - 0.1),
            min(a.segment.end, vc.duration),
        )

        if subclip.duration < slot_dur:
            subclip = subclip.loop(duration=slot_dur)
        else:
            subclip = subclip.subclip(0, slot_dur)

        # Speed ramp: slow down around the kill moment if known
        if apply_speed_ramp and a.segment.kill_time is not None:
            kill_in_clip = a.segment.kill_time - a.segment.start
            if 0 < kill_in_clip < subclip.duration:
                subclip = speed_ramp(subclip, kill_time=kill_in_clip)

        if apply_color_grade:
            subclip = color_grade(subclip)

        subclip = subclip.resize(resolution)

        if gameplay_volume == 0.0:
            subclip = subclip.without_audio()
        elif gameplay_volume < 1.0 and subclip.audio is not None:
            subclip = subclip.set_audio(subclip.audio.volumex(gameplay_volume))

        video_segments.append(subclip)

    # Concatenate with transitions
    if transition == TransitionType.HARD_CUT or len(video_segments) == 1:
        final_video = concatenate_videoclips(video_segments, method="compose")
    else:
        combined = video_segments[0]
        for next_seg in video_segments[1:]:
            combined = apply_transition(combined, next_seg, transition)
        final_video = combined

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
