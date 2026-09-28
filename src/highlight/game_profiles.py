"""Per-game tuning profiles."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameProfile:
    name: str
    # Kill feed region as fraction of frame: (x, y, w, h), or None if game has no useful ROI
    kill_feed_roi: tuple[float, float, float, float] | None
    audio_weight: float
    motion_weight: float
    # Percentile above which a beat is considered a peak
    peak_percentile: float = 75.0


_PROFILES: dict[str, GameProfile] = {
    "csgo": GameProfile(
        name="csgo",
        kill_feed_roi=(0.72, 0.0, 0.28, 0.25),   # top-right
        audio_weight=0.65,   # gunshots are loud and distinct
        motion_weight=0.35,
    ),
    "cs2": GameProfile(
        name="cs2",
        kill_feed_roi=(0.72, 0.0, 0.28, 0.25),
        audio_weight=0.65,
        motion_weight=0.35,
    ),
    "valorant": GameProfile(
        name="valorant",
        kill_feed_roi=(0.72, 0.0, 0.28, 0.20),   # top-right, slightly shorter
        audio_weight=0.5,
        motion_weight=0.5,
    ),
    "league": GameProfile(
        name="league",
        kill_feed_roi=None,   # LoL kill feed is top-center, less reliable
        audio_weight=0.35,
        motion_weight=0.65,   # spell animations drive action more than audio
    ),
    "lol": GameProfile(
        name="league",
        kill_feed_roi=None,
        audio_weight=0.35,
        motion_weight=0.65,
    ),
    "generic": GameProfile(
        name="generic",
        kill_feed_roi=None,
        audio_weight=0.5,
        motion_weight=0.5,
    ),
}

_ALIASES = {
    "counter-strike": "csgo",
    "counterstrike": "csgo",
    "leagueoflegends": "league",
    "league of legends": "league",
    "val": "valorant",
}


def get_profile(game: str) -> GameProfile:
    key = game.lower().strip()
    key = _ALIASES.get(key, key)
    if key not in _PROFILES:
        print(f"  Unknown game '{game}', using generic profile.")
        return _PROFILES["generic"]
    return _PROFILES[key]
