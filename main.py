"""
fps-mv  —  Auto highlight editor for FPS/MOBA gameplay footage.

Usage examples:

  # Basic: one music track + multiple clips
  python main.py --music track.mp3 clip1.mp4 clip2.mp4 clip3.mp4

  # CS:GO with kill-feed detection, flash-white transitions, 90s max
  python main.py --music track.mp3 --game csgo --duration 90 clips/*.mp4

  # Quick preview draft (480p, no effects, fast)
  python main.py --music track.mp3 --game valorant --preview clips/*.mp4

  # Keep some gameplay audio, whip-zoom transitions, no color grade
  python main.py --music track.mp3 --gameplay-vol 0.15 --transition whip-zoom --no-color-grade clips/*.mp4
"""

import sys
from pathlib import Path

import click

from src.utils.ffmpeg_check import require_ffmpeg
from src.ingestion.loader import load_clips
from src.highlight.scorer import score_clip
from src.highlight.game_profiles import get_profile
from src.highlight.kill_detector import detect_kill_events
from src.audio.analyzer import analyze_music
from src.editor.assembler import plan_edit, render
from src.editor.transitions import TransitionType
from src.utils.visualizer import print_edit_plan


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("clips", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--music", "-m", required=True, type=click.Path(exists=True),
              help="Music audio file (mp3/wav/flac/ogg)")
@click.option("--output", "-o", default="output/mv.mp4", show_default=True,
              help="Output video path")
@click.option("--game", "-g", default="generic", show_default=True,
              help="Game profile: csgo | cs2 | valorant | league | generic")
@click.option("--duration", "-d", default=None, type=float,
              help="Max output duration in seconds (defaults to full music length)")
@click.option("--threshold", default=0.55, show_default=True,
              help="Highlight score threshold 0–1 (lower = more segments found)")
@click.option("--window", default=3.0, show_default=True,
              help="Highlight segment length in seconds")
@click.option("--gameplay-vol", default=0.0, show_default=True,
              help="Gameplay audio volume in final mix (0=mute, 1=full)")
@click.option("--music-vol", default=1.0, show_default=True,
              help="Music volume in final mix")
@click.option("--fps", default=60, show_default=True, help="Output FPS")
@click.option("--resolution", default="1920x1080", show_default=True,
              help="Output resolution WxH")
@click.option("--peak-percentile", default=75.0, show_default=True,
              help="Energy percentile to classify a beat as a 'peak' (drop/chorus)")
@click.option("--transition", default="flash-white", show_default=True,
              type=click.Choice(["cut", "flash-white", "flash-black", "whip-zoom"]),
              help="Transition style between clips")
@click.option("--color-grade/--no-color-grade", default=True, show_default=True,
              help="Apply saturation/contrast boost to each clip")
@click.option("--speed-ramp/--no-speed-ramp", default=True, show_default=True,
              help="Apply slow-mo at kill moments when detected")
@click.option("--preview", is_flag=True, default=False,
              help="Fast low-res draft render (480p, 24fps, hard cuts, no effects)")
def main(
    clips, music, output, game, duration, threshold, window,
    gameplay_vol, music_vol, fps, resolution, peak_percentile,
    transition, color_grade, speed_ramp, preview,
):
    """Auto-edit FPS/MOBA gameplay clips synced to a music track."""
    require_ffmpeg()

    w, h = (int(x) for x in resolution.lower().split("x"))
    profile = get_profile(game)
    trans = TransitionType(transition)

    print(f"\nfps-mv  |  {len(clips)} clip(s)  |  game: {profile.name}  |  music: {Path(music).name}\n")

    # 1. Load clips
    print("Loading clips...")
    clip_objs = load_clips(list(clips))
    for c in clip_objs:
        print(f"  {c.path.name}  ({c.duration:.1f}s  {c.width}x{c.height}  {c.fps:.0f}fps)")

    # 2. Detect kill events (game-specific ROI) then score highlights
    print("\nDetecting highlights...")
    for c in clip_objs:
        kill_times = detect_kill_events(c.path, profile)
        if kill_times:
            print(f"  Kill events detected in {c.path.name}: {len(kill_times)} @ "
                  + ", ".join(f"{t:.1f}s" for t in kill_times[:5])
                  + (" ..." if len(kill_times) > 5 else ""))

        c.highlight_segments = score_clip(
            c.path,
            audio_weight=profile.audio_weight,
            motion_weight=profile.motion_weight,
            window_sec=window,
            threshold=threshold,
            kill_event_times=kill_times if kill_times else None,
        )
        print(f"  → {len(c.highlight_segments)} segment(s) found in {c.path.name}")

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
    print_edit_plan(assignments, music_analysis)

    # 5. Render
    out_path = output
    if preview and not out_path.endswith("_preview.mp4"):
        stem = Path(output).stem
        out_path = str(Path(output).parent / f"{stem}_preview.mp4")

    render(
        assignments,
        music_analysis,
        output_path=out_path,
        resolution=(w, h),
        fps=fps,
        gameplay_volume=gameplay_vol,
        music_volume=music_vol,
        transition=trans,
        apply_color_grade=color_grade,
        apply_speed_ramp=speed_ramp,
        preview=preview,
    )


if __name__ == "__main__":
    main()
