"""Shared data model (single source of truth). Owned by the lead – ask before changing."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json

SCHEMA_VERSION = 1


@dataclass
class Section:
    start_bar: int
    end_bar: int            # exclusive
    kind: str               # intro | verse | build | drop | break | outro


@dataclass
class TrackAnalysis:
    path: str                       # original file
    file_hash: str                  # sha1 of file bytes (cache key)
    title: str
    artist: str
    duration: float                 # seconds
    sample_rate: int                # of the decoded cache audio (44100)
    bpm: float                      # median/fitted tempo
    constant_tempo: bool
    beats: List[float]              # all beat times in seconds (source time)
    first_downbeat_index: int       # index in `beats` of bar 0
    key: str                        # Camelot, e.g. "8A"; "" if unknown
    key_confidence: float
    phrase_offset: int              # bar index 0..7 where 8-bar phrases start
    # per-bar features, one entry per bar starting at bar 0 (dB values)
    bar_bass: List[float] = field(default_factory=list)
    bar_mid: List[float] = field(default_factory=list)
    bar_high: List[float] = field(default_factory=list)
    bar_vocal: List[float] = field(default_factory=list)
    bar_rms: List[float] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    energy: float = 5.0             # 1..10
    loudness_lufs: float = -14.0    # integrated loudness of the loud part
    cache_dir: str = ""             # contains audio.flac, inst.flac, voc.flac
    schema: int = SCHEMA_VERSION

    @property
    def n_bars(self) -> int:
        return (len(self.beats) - self.first_downbeat_index) // 4

    def bar_time(self, bar: float) -> float:
        """Source time (s) of a fractional bar index via the beat list (extrapolates at the ends)."""
        b = self.beats
        x = self.first_downbeat_index + bar * 4
        i = int(min(max(int(x // 1), 0), len(b) - 2))
        return b[i] + (x - i) * (b[i + 1] - b[i])

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TrackAnalysis":
        d = json.loads(s)
        d["sections"] = [Section(**x) for x in d.get("sections", [])]
        return TrackAnalysis(**d)


@dataclass
class Transition:
    overlap_bars: int               # 0 = cut/echo, else 4/8/16
    tempo: float                    # BPM held constant during the overlap
    preset: str                     # xfade | swap | swap_hp | editor | echo_out | echo_cut | double
    options: Dict[str, Any] = field(default_factory=dict)
    # options: swap (0..1 fraction where bass+vocals hand over), fadein (0..1), lpin (bool),
    #          lpdur (0..1), voc (bool, vocal hand-over on – default True),
    #          tempo_in (float, only for overlap 0 / echo_cut: tempo of the incoming track right
    #          after the cut; the layout jumps from `tempo` to `tempo_in` at the cut)


@dataclass
class PlannedTrack:
    track_hash: str                 # TrackAnalysis.file_hash
    start_bar: int                  # source bar where playback starts
    end_bar: int                    # source bar where it ends (exclusive); next track's overlap ends here


@dataclass
class MixPlan:
    tracks: List[PlannedTrack]
    transitions: List[Transition]   # len = len(tracks) - 1
    start_tempo: float
    end_tempo: float
    end_fade_bars: int = 8
    target_minutes: float = 30.0
    notes: List[str] = field(default_factory=list)   # Dutch explanation per transition (GUI)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(s: str) -> "MixPlan":
        d = json.loads(s)
        return MixPlan(tracks=[PlannedTrack(**t) for t in d["tracks"]],
                       transitions=[Transition(**t) for t in d["transitions"]],
                       **{k: v for k, v in d.items() if k not in ("tracks", "transitions")})


@dataclass
class RenderSettings:
    out_path: str
    formats: List[str] = field(default_factory=lambda: ["flac"])   # flac | wav | mp3
    bit_depth: int = 24
    target_lufs: float = -9.5
    true_peak_db: float = -1.0
    verify: bool = False
