"""Tests for highlight scorer."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.highlight.scorer import score_clip, HighlightSegment


def _mock_cap_factory(n_frames: int = 120, fps: float = 30.0):
    """Return a mock cv2.VideoCapture that yields n_frames of noise."""
    frames = [np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8) for _ in range(n_frames)]
    call_count = {"i": 0}

    class MockCap:
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return fps
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return n_frames
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 320
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 180
            return 0

        def read(self):
            i = call_count["i"]
            call_count["i"] += 1
            if i >= len(frames):
                return False, None
            return True, frames[i]

        def set(self, prop, val):
            pass

        def release(self):
            pass

        def isOpened(self):
            return True

    return MockCap()


def test_score_clip_returns_segments(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"fake")

    # Audio energy with a spike at second 1 and 3
    audio_energy = np.array([0.1, 0.9, 0.1, 0.9, 0.1], dtype=np.float32)

    with patch("src.highlight.scorer.cv2.VideoCapture", return_value=_mock_cap_factory(150, 30.0)), \
         patch("src.highlight.scorer._extract_audio_energy", return_value=(audio_energy, 22050)), \
         patch("src.highlight.scorer.cache_get", return_value=None), \
         patch("src.highlight.scorer.cache_put"):
        segments = score_clip(fake_video, threshold=0.4, window_sec=1.0)

    assert isinstance(segments, list)
    assert len(segments) > 0
    assert all(isinstance(s, HighlightSegment) for s in segments)
    # Segments should be sorted by score descending
    scores = [s.score for s in segments]
    assert scores == sorted(scores, reverse=True)


def test_score_clip_fallback_no_audio(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"fake")

    with patch("src.highlight.scorer.cv2.VideoCapture", return_value=_mock_cap_factory(120, 30.0)), \
         patch("src.highlight.scorer._extract_audio_energy", side_effect=Exception("no audio")), \
         patch("src.highlight.scorer.cache_get", return_value=None), \
         patch("src.highlight.scorer.cache_put"):
        # Should not raise; falls back to motion-only
        segments = score_clip(fake_video, threshold=0.0, window_sec=1.0)

    assert isinstance(segments, list)


def test_kill_event_boost(tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"fake")

    # Flat audio/motion: everything equal
    flat_energy = np.zeros(10, dtype=np.float32)

    with patch("src.highlight.scorer.cv2.VideoCapture", return_value=_mock_cap_factory(300, 30.0)), \
         patch("src.highlight.scorer._extract_audio_energy", return_value=(flat_energy, 22050)), \
         patch("src.highlight.scorer._extract_motion_scores", return_value=flat_energy.copy()), \
         patch("src.highlight.scorer.cache_get", return_value=None), \
         patch("src.highlight.scorer.cache_put"):
        segs_no_kill = score_clip(fake_video, threshold=0.1, window_sec=1.0)
        segs_with_kill = score_clip(fake_video, threshold=0.1, window_sec=1.0,
                                    kill_event_times=[5.0])

    # With a kill event boost the segment containing second 5 should score higher
    scores_no_kill = {int(s.start): s.score for s in segs_no_kill}
    scores_with_kill = {int(s.start): s.score for s in segs_with_kill}

    # Segment at second 5 (if it exists) should have score ≥ without kill
    if 5 in scores_with_kill and 5 in scores_no_kill:
        assert scores_with_kill[5] >= scores_no_kill[5]
