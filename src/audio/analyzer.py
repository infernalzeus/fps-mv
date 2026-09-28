"""
Music analysis: extract beats, tempo, and energy peaks from a music track.

Beat times are used as cut points. Energy peaks (drops, chorus hits) are
prioritized as sync anchors — clips start on these for maximum impact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa


@dataclass
class MusicAnalysis:
    path: Path
    duration: float
    tempo: float
    beat_times: np.ndarray      # seconds of each beat
    peak_times: np.ndarray      # seconds of high-energy moments (drops/chorus)
    peak_intensities: np.ndarray  # normalized energy at each peak


def analyze_music(music_path: str | Path, peak_percentile: float = 75.0) -> MusicAnalysis:
    """
    Load a music file and extract rhythmic structure.

    peak_percentile : energy threshold above which a beat is considered a peak.
                      75 = top quarter of the energy distribution.
    """
    path = Path(music_path)
    if not path.exists():
        raise FileNotFoundError(f"Music file not found: {music_path}")

    print(f"Analyzing music: {path.name}")
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Per-beat RMS energy
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    beat_rms = np.array([
        rms[min(librosa.time_to_frames(t, sr=sr, hop_length=hop), len(rms) - 1)]
        for t in beat_times
    ], dtype=np.float32)

    # Normalize
    rms_min, rms_max = beat_rms.min(), beat_rms.max()
    beat_rms_norm = (beat_rms - rms_min) / (rms_max - rms_min + 1e-8)

    threshold = np.percentile(beat_rms_norm, peak_percentile)
    peak_mask = beat_rms_norm >= threshold
    peak_times = beat_times[peak_mask]
    peak_intensities = beat_rms_norm[peak_mask]

    tempo_val = float(np.atleast_1d(tempo)[0])
    print(f"  Tempo: {tempo_val:.1f} BPM | Beats: {len(beat_times)} | Peaks: {len(peak_times)}")

    return MusicAnalysis(
        path=path,
        duration=duration,
        tempo=tempo_val,
        beat_times=beat_times,
        peak_times=peak_times,
        peak_intensities=peak_intensities,
    )
