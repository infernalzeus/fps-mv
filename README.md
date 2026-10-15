# fps-mv

Auto-edits FPS/MOBA gameplay clips into a highlight "music video", cut to the beat of a track you choose.

Give it a pile of clips and one song. It finds the exciting moments in each clip, finds the beats and
peaks in the song, then lays the highlights onto the beat grid with transitions and renders the result.

## How it works

```
clips ──► load ──► score highlights ──► ┐
                   (audio energy +      │
                    motion + kill feed) ├──► plan edit ──► render (ffmpeg)
music ──► analyse beats / peaks ───────► ┘
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `src/ingestion/loader.py` | Reads each clip's duration, resolution, fps |
| Kill events | `src/highlight/kill_detector.py` | Watches the game's kill-feed region for new entries |
| Highlight scoring | `src/highlight/scorer.py` | Combines audio energy and frame motion (0–1) in sliding windows; keeps windows above `--threshold` |
| Game tuning | `src/highlight/game_profiles.py` | Per-game audio/motion weights and kill-feed region |
| Music | `src/audio/analyzer.py` | Beat grid and "peak" beats (drops/choruses) via librosa |
| Edit plan | `src/editor/assembler.py` | Assigns highlight segments to beat slots, up to `--duration` |
| Effects | `src/editor/transitions.py`, `effects.py` | Flash/whip transitions, colour grade, slow-mo at kills |

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
pip install -r requirements.txt
```

## Use

```bash
# one track, several clips
python main.py --music track.mp3 clip1.mp4 clip2.mp4 clip3.mp4

# CS2 kill-feed detection, 90 s cap
python main.py --music track.mp3 --game cs2 --duration 90 clips/*.mp4

# quick low-res draft (480p, hard cuts, no effects)
python main.py --music track.mp3 --game valorant --preview clips/*.mp4

# keep a little gameplay audio, whip-zoom transitions, no colour grade
python main.py --music track.mp3 --gameplay-vol 0.15 --transition whip-zoom --no-color-grade clips/*.mp4
```

Output goes to `output/mv.mp4` by default.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--music, -m` | required | Music file (mp3/wav/flac/ogg) |
| `--output, -o` | `output/mv.mp4` | Output path |
| `--game, -g` | `generic` | `csgo`, `cs2`, `valorant`, `league`, `generic` |
| `--duration, -d` | music length | Max output length (s) |
| `--threshold` | `0.55` | Highlight score cut-off; lower finds more segments |
| `--window` | `3.0` | Highlight segment length (s) |
| `--gameplay-vol` / `--music-vol` | `0.0` / `1.0` | Mix levels |
| `--fps` / `--resolution` | `60` / `1920x1080` | Output format |
| `--peak-percentile` | `75` | Energy percentile that counts as a peak beat |
| `--transition` | `flash-white` | `cut`, `flash-white`, `flash-black`, `whip-zoom` |
| `--color-grade/--no-color-grade` | on | Saturation/contrast boost |
| `--speed-ramp/--no-speed-ramp` | on | Slow-mo at detected kills |
| `--preview` | off | Fast low-res draft |

If no highlights are found, lower `--threshold` (e.g. `0.35`).

## Tests

```bash
pytest
```

Covers the music analyser, highlight scorer, edit assembler and cache.
