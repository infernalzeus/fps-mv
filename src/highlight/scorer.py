"""
Scores gameplay clips to find highlight moments.

Uses two signals:
  1. Audio energy spikes  — gunshots, ability sounds, crowd reaction
  2. Frame motion         — optical-flow magnitude between frames

Each signal is normalized to [0, 1] and combined with configurable weights.
The result is a per-second score array; we then extract windows above a threshold.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path

from src.utils.cache import make_key, cache_get, cache_put


@dataclass
class HighlightSegment:
    start: float            # seconds
    end: float              # seconds
    score: float            # 0–1 combined highlight score
    kill_time: float | None = None  # first kill event within this segment (seconds, absolute)


def _extract_audio_energy(video_path: Path, sr: int = 22050) -> tuple[np.ndarray, float]:
    """Return per-second RMS energy array extracted from the video's audio track."""
    import subprocess, tempfile, librosa

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
         "-ar", str(sr), tmp_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    y, _ = librosa.load(tmp_path, sr=sr, mono=True)
    import os; os.unlink(tmp_path)

    # RMS in 1-second windows
    frame_len = sr
    hop = sr
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    return rms.astype(np.float32), sr


def _extract_motion_scores(video_path: Path, sample_fps: float = 4.0) -> np.ndarray:
    """Return per-second mean optical-flow magnitude."""
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0

    step = max(1, int(native_fps / sample_fps))
    n_seconds = max(1, int(duration))
    per_second = np.zeros(n_seconds, dtype=np.float32)
    counts = np.zeros(n_seconds, dtype=np.int32)

    prev_gray = None
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
                t = int(frame_idx / native_fps)
                if t < n_seconds:
                    per_second[t] += mag
                    counts[t] += 1
            prev_gray = gray
        frame_idx += 1

    cap.release()
    mask = counts > 0
    per_second[mask] /= counts[mask]
    return per_second


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def score_clip(
    video_path: Path,
    audio_weight: float = 0.5,
    motion_weight: float = 0.5,
    window_sec: float = 3.0,
    threshold: float = 0.55,
    min_gap_sec: float = 2.0,
    kill_event_times: list[float] | None = None,
) -> list[HighlightSegment]:
    """
    Analyze a gameplay clip and return a list of highlight segments.

    Parameters
    ----------
    audio_weight / motion_weight : relative contribution of each signal (sum to 1)
    window_sec : length of each highlight segment in seconds
    threshold  : minimum combined score (0–1) to qualify as a highlight
    min_gap_sec: merge segments closer than this
    kill_event_times: optional per-second kill event timestamps; each one boosts score by 0.3
    """
    params = dict(
        audio_weight=audio_weight, motion_weight=motion_weight,
        window_sec=window_sec, threshold=threshold, min_gap_sec=min_gap_sec,
    )
    cache_key = make_key(video_path, params)
    cached = cache_get(cache_key)
    if cached is not None:
        print(f"  Scoring (cache hit): {video_path.name}")
        return [
            HighlightSegment(start=s["start"], end=s["end"], score=s["score"],
                             kill_time=s.get("kill_time"))
            for s in cached
        ]

    print(f"  Scoring: {video_path.name}")

    try:
        audio_energy, _ = _extract_audio_energy(video_path)
        has_audio = True
    except Exception:
        has_audio = False
        audio_energy = np.array([], dtype=np.float32)

    motion = _extract_motion_scores(video_path)
    n_sec = len(motion)

    if has_audio and len(audio_energy) > 0:
        min_len = min(n_sec, len(audio_energy))
        motion_norm = _normalize(motion[:min_len])
        audio_norm = _normalize(audio_energy[:min_len])
        combined = audio_weight * audio_norm + motion_weight * motion_norm
    else:
        combined = _normalize(motion)

    # Boost seconds where kill events were detected
    if kill_event_times:
        for t in kill_event_times:
            idx = int(t)
            if 0 <= idx < len(combined):
                combined[idx] = min(1.0, combined[idx] + 0.3)

    # Sliding window scoring
    win = max(1, int(window_sec))
    segments: list[HighlightSegment] = []
    last_end = -min_gap_sec

    for i in range(len(combined) - win + 1):
        score = float(combined[i : i + win].mean())
        if score >= threshold:
            start = float(i)
            end = float(i + win)
            if start >= last_end + min_gap_sec:
                segments.append(HighlightSegment(start=start, end=end, score=score))
                last_end = end

    # Annotate segments with the first kill event that falls within them
    if kill_event_times:
        for seg in segments:
            for kt in kill_event_times:
                if seg.start <= kt <= seg.end:
                    seg.kill_time = kt
                    break

    segments.sort(key=lambda s: s.score, reverse=True)
    cache_put(cache_key, [
        {"start": s.start, "end": s.end, "score": s.score, "kill_time": s.kill_time}
        for s in segments
    ])
    return segments
