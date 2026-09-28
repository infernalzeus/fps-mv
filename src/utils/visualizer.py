"""ASCII timeline visualizer for the edit plan."""
from __future__ import annotations

from src.audio.analyzer import MusicAnalysis


def print_edit_plan(assignments: list, music: MusicAnalysis) -> None:
    """Print an ASCII timeline of the edit plan to stdout."""
    if not assignments:
        print("  (no assignments)")
        return

    peak_set = set(music.peak_times.round(3))
    total_dur = assignments[-1].music_end if assignments else 0.0

    print(f"\n{'─' * 72}")
    print(f"  EDIT PLAN  |  {len(assignments)} clips  |  {_fmt(total_dur)} total")
    print(f"{'─' * 72}")

    for a in assignments:
        minutes = int(a.music_start // 60)
        seconds = a.music_start % 60
        timestamp = f"{minutes:02d}:{seconds:05.2f}"

        score = a.segment.score
        bar_filled = int(score * 8)
        bar = "█" * bar_filled + "░" * (8 - bar_filled)

        is_peak = round(a.music_start, 3) in peak_set
        peak_tag = " ★ PEAK" if is_peak else ""
        kill_tag = " ⚡ KILL" if a.segment.kill_time is not None else ""

        clip_name = a.clip.path.name[:24].ljust(24)
        seg_info = f"{a.segment.end - a.segment.start:.1f}s@{_fmt(a.segment.start)}"

        print(f"  {timestamp}  {bar}  {clip_name}  [{seg_info}]  {score:.2f}{peak_tag}{kill_tag}")

    print(f"{'─' * 72}\n")


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:04.1f}"
