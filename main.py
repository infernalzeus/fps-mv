"""
fps-mv  —  Auto highlight editor for FPS/MOBA gameplay footage.

Usage examples:

  # Basic: one music track + multiple clips, output to ./output/mv.mp4
  python main.py --music track.mp3 clip1.mp4 clip2.mp4 clip3.mp4

  # Keep 90 seconds max, cap output resolution at 1080p
  python main.py --music track.mp3 --duration 90 --output out/frag_movie.mp4 clips/*.mp4

  # Keep some gameplay audio audible (20%)
  python main.py --music track.mp3 --gameplay-vol 0.2 clips/*.mp4

  # Tune highlight sensitivity (lower threshold = more lenient, more segments)
  python main.py --music track.mp3 --threshold 0.45 clips/*.mp4
"""

import sys
from pathlib import Path

import click

from src.utils.ffmpeg_check import require_ffmpeg
from src.ingestion.loader import load_clips
from src.highlight.scorer import score_clip
from src.audio.analyzer import analyze_music
from src.editor.assembler import plan_edit, render


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("clips", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--music", "-m", required=True, type=click.Path(exists=True),
              help="Music audio file (mp3/wav/flac/ogg)")
@click.option("--output", "-o", default="output/mv.mp4", show_default=True,
              help="Output video path")
@click.option("--duration", "-d", default=None, type=float,
              help="Max output duration in seconds (defaults to full music length)")
@click.option("--threshold", default=0.55, show_default=True,
              help="Highlight score threshold 0–1 (lower = more segments found)")
@click.option("--window", default=3.0, show_default=True,
              help="Highlight segment length in seconds")
@click.option("--audio-weight", default=0.5, show_default=True,
              help="Weight for gameplay audio energy signal (0–1)")
@click.option("--motion-weight", default=0.5, show_default=True,
              help="Weight for motion/action signal (0–1)")
@click.option("--gameplay-vol", default=0.0, show_default=True,
              help="Gameplay audio volume in final mix (0=mute, 1=full)")
@click.option("--music-vol", default=1.0, show_default=True,
              help="Music volume in final mix")
@click.option("--fps", default=60, show_default=True, help="Output FPS")
@click.option("--resolution", default="1920x1080", show_default=True,
              help="Output resolution WxH")
@click.option("--peak-percentile", default=75.0, show_default=True,
              help="Energy percentile to classify a beat as a 'peak' (drop/chorus)")
def main(
    clips, music, output, duration, threshold, window,
    audio_weight, motion_weight, gameplay_vol, music_vol,
    fps, resolution, peak_percentile,
):
    """Auto-edit FPS/MOBA gameplay clips synced to a music track."""
    require_ffmpeg()

    w, h = (int(x) for x in resolution.lower().split("x"))

    print(f"\nfps-mv  |  {len(clips)} clip(s)  |  music: {Path(music).name}\n")

    # 1. Load clips
    print("Loading clips...")
    clip_objs = load_clips(list(clips))
    for c in clip_objs:
        print(f"  {c.path.name}  ({c.duration:.1f}s  {c.width}x{c.height}  {c.fps:.0f}fps)")

    # 2. Detect highlights in each clip
    print("\nDetecting highlights...")
    for c in clip_objs:
        c.highlight_segments = score_clip(
            c.path,
            audio_weight=audio_weight,
            motion_weight=motion_weight,
            window_sec=window,
            threshold=threshold,
        )
        print(f"  → {len(c.highlight_segments)} segment(s) found")

    total_segments = sum(len(c.highlight_segments) for c in clip_objs)
    if total_segments == 0:
        click.echo(
            "\nNo highlight segments found. Try lowering --threshold (e.g. --threshold 0.35).",
            err=True,
        )
        sys.exit(1)

    # 3. Analyze music
    print("\nAnalyzing music...")
    music_analysis = analyze_music(music, peak_percentile=peak_percentile)

    # 4. Plan edit
    print("\nPlanning edit...")
    assignments = plan_edit(clip_objs, music_analysis, max_duration=duration)
    print(f"  → {len(assignments)} beat slots assigned")

    # 5. Render
    render(
        assignments,
        music_analysis,
        output_path=output,
        resolution=(w, h),
        fps=fps,
        gameplay_volume=gameplay_vol,
        music_volume=music_vol,
    )


if __name__ == "__main__":
    main()
