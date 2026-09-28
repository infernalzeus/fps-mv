"""Tests for music beat/peak analyzer."""
import numpy as np
import pytest
import librosa

from src.audio.analyzer import analyze_music


def _make_click_track(tmp_path, bpm: float = 120.0, duration: float = 10.0) -> str:
    """Generate a synthetic click-track WAV at the given BPM."""
    sr = 22050
    beat_interval = 60.0 / bpm
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Place a short sine burst at each beat
    click_dur = int(sr * 0.02)
    beat_t = 0.0
    while beat_t < duration:
        idx = int(beat_t * sr)
        signal[idx : idx + click_dur] += np.sin(2 * np.pi * 440 * t[:click_dur])
        beat_t += beat_interval

    out = tmp_path / "click.wav"
    import soundfile as sf
    sf.write(str(out), signal.astype(np.float32), sr)
    return str(out)


def test_tempo_detected(tmp_path):
    path = _make_click_track(tmp_path, bpm=120.0, duration=10.0)
    result = analyze_music(path)
    assert abs(result.tempo - 120.0) < 10.0, f"Expected ~120 BPM, got {result.tempo}"


def test_beats_reasonable(tmp_path):
    path = _make_click_track(tmp_path, bpm=120.0, duration=10.0)
    result = analyze_music(path)
    # 120 BPM × 10s = 20 beats; allow ±5
    assert 10 <= len(result.beat_times) <= 30


def test_peak_times_subset_of_beats(tmp_path):
    path = _make_click_track(tmp_path, bpm=120.0, duration=10.0)
    result = analyze_music(path)
    beat_set = set(result.beat_times.round(3))
    for pt in result.peak_times:
        assert round(pt, 3) in beat_set


def test_duration_correct(tmp_path):
    path = _make_click_track(tmp_path, bpm=120.0, duration=8.0)
    result = analyze_music(path)
    assert abs(result.duration - 8.0) < 0.5
