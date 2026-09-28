"""Tests for plan_edit: temporal order, peak assignment, diversity."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.loader import Clip
from src.highlight.scorer import HighlightSegment
from src.audio.analyzer import MusicAnalysis
from src.editor.assembler import plan_edit


def _make_clip(name: str, segments: list[HighlightSegment]) -> Clip:
    c = MagicMock(spec=Clip)
    c.path = Path(f"/fake/{name}.mp4")
    c.highlight_segments = segments
    c.duration = 60.0
    return c


def _make_music(n_beats: int = 10, bpm: float = 120.0, peak_indices: list[int] | None = None) -> MusicAnalysis:
    beat_interval = 60.0 / bpm
    beat_times = np.array([i * beat_interval for i in range(n_beats)])
    peak_idx = peak_indices or [0, 4, 8]
    peak_times = beat_times[[i for i in peak_idx if i < n_beats]]
    return MusicAnalysis(
        path=Path("/fake/music.mp3"),
        duration=beat_times[-1] + beat_interval,
        tempo=bpm,
        beat_times=beat_times,
        peak_times=peak_times,
        peak_intensities=np.ones(len(peak_times)),
    )


def test_temporal_order():
    clips = [
        _make_clip("a", [HighlightSegment(0.0, 3.0, 0.9), HighlightSegment(5.0, 8.0, 0.7)]),
        _make_clip("b", [HighlightSegment(10.0, 13.0, 0.6)]),
    ]
    music = _make_music(n_beats=10)
    assignments = plan_edit(clips, music)

    # Assignments must be sorted by music_start (temporal order)
    starts = [a.music_start for a in assignments]
    assert starts == sorted(starts), "Assignments not in temporal order"


def test_peak_slots_get_best_clips():
    high_seg = HighlightSegment(0.0, 3.0, 0.95)
    low_seg = HighlightSegment(5.0, 8.0, 0.30)
    clips = [_make_clip("a", [high_seg, low_seg])]
    music = _make_music(n_beats=6, peak_indices=[0])  # slot 0 is a peak

    assignments = plan_edit(clips, music)

    # The assignment at slot 0 (the peak) should have the highest-score segment
    peak_assignment = next(a for a in assignments if a.music_start < 0.6)
    assert peak_assignment.segment.score >= 0.9, \
        f"Peak slot should get best clip, got score {peak_assignment.segment.score}"


def test_no_consecutive_same_clip():
    # Two clips with several segments each — diversity is achievable
    segs_a = [HighlightSegment(float(i * 5), float(i * 5 + 3), 0.8 - i * 0.05) for i in range(6)]
    segs_b = [HighlightSegment(float(i * 5), float(i * 5 + 3), 0.75 - i * 0.05) for i in range(6)]
    clips = [_make_clip("clip_a", segs_a), _make_clip("clip_b", segs_b)]
    music = _make_music(n_beats=10)

    assignments = plan_edit(clips, music)

    # With two clips, same clip should never appear in consecutive slots
    for i in range(len(assignments) - 1):
        a, b = assignments[i], assignments[i + 1]
        assert not (a.clip.path == b.clip.path), \
            f"Same clip used in consecutive slots at positions {i} and {i+1}"


def test_no_segment_repeat_within_pool():
    # 5 unique segments, 5 slots — no repeats should occur at all
    segs = [HighlightSegment(float(i * 5), float(i * 5 + 3), 0.8 - i * 0.02) for i in range(5)]
    clips = [_make_clip("a", segs)]
    music = _make_music(n_beats=6)  # 5 slots

    assignments = plan_edit(clips, music)
    assert len(assignments) == 5

    seen_pairs = set()
    for a in assignments:
        pair = (str(a.clip.path), a.segment.start)
        assert pair not in seen_pairs, f"Segment {pair} used more than once"
        seen_pairs.add(pair)


def test_multiple_clips_diversity():
    clips = [
        _make_clip("clip1", [HighlightSegment(0.0, 3.0, 0.85)]),
        _make_clip("clip2", [HighlightSegment(0.0, 3.0, 0.80)]),
        _make_clip("clip3", [HighlightSegment(0.0, 3.0, 0.75)]),
    ]
    music = _make_music(n_beats=8)
    assignments = plan_edit(clips, music)

    # Should use at least 2 different clips
    used_clips = {str(a.clip.path) for a in assignments}
    assert len(used_clips) >= 2
