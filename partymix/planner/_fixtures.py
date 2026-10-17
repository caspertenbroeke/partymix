"""Test fixture: TrackAnalysis objects for the 14 real tracks of the hand-made "Half uur mix v2".

Built from reference/bars.json (beats, first downbeat, per-bar bass/mid/high dB), reference/grid2.json
(bpm, duration) and ``_fixture_audio.json`` (per-bar vocal dB and mix RMS dB computed once from the
separated stems; falls back to the vocal digits in reference/vox.json). Phrase offset, sections and
energy use simple heuristics – the real ones come from partymix.analysis.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..model import Section, TrackAnalysis

REF = Path(__file__).resolve().parents[2] / "reference"
AUDIO = Path(__file__).resolve().parent / "_fixture_audio.json"

# key: (title, artist, camelot)
META: Dict[str, tuple] = {
    "01": ("World Hold On", "Bob Sinclar", "1A"),
    "02": ("Stereo Love", "Edward Maya", "12A"),
    "03": ("Die Young", "Kesha", "12B"),
    "04": ("Levels", "Avicii", "12B"),
    "05": ("Rampeneren", "", "1B"),
    "06": ("It's That Time", "", "1A"),
    "06m": ("Memories", "David Guetta", "3A"),
    "07": ("Kaolo Beat", "", "3B"),
    "08": ("Y Que Fue", "", "1A"),
    "09": ("La Mama", "", "5A"),
    "10": ("Schudden", "", "8A"),
    "11": ("Brokko Maken", "", "3A"),
    "12": ("Axel F", "Crazy Frog", "10B"),
    "13": ("We Like To Party", "Vengaboys", "4B"),
}
V2_ORDER = ["01", "02", "03", "04", "05", "06", "06m", "07", "08", "09", "10", "11", "12", "13"]


def _levels(x: np.ndarray, span: float, step: float) -> np.ndarray:
    top = np.percentile(x, 90)
    return np.clip((x - top + span) / step, 0, 9)


def phrase_offset_heuristic(bass_db: List[float], vocal_db: List[float]) -> int:
    """Bar 0..7 where 8-bar phrases start: maximise bass/vocal novelty on the 8-bar grid."""
    b = _levels(np.asarray(bass_db, float), 18, 2)
    v = _levels(np.asarray(vocal_db, float), 20, 2.2) if len(vocal_db) else np.zeros_like(b)
    n = len(b)

    def nov(w):
        x = np.zeros(n)
        for i in range(w, n - w + 1):
            x[i] = abs(b[i:i + w].mean() - b[i - w:i].mean()) + 0.7 * abs(v[i:i + w].mean() - v[i - w:i].mean())
        return x
    n1, n4 = nov(1), nov(4)
    # 4-bar phase from single-bar changes (prior: tracks usually start on a phrase), then 0 vs +4
    s4 = [n1[o::4].sum() * (1.4 if o == 0 else 1.0) for o in range(4)]
    p = int(np.argmax(s4))
    return p + 4 if n4[p + 4::8].sum() > n4[p::8].sum() else p


def sections_heuristic(bass_db: List[float], vocal_db: List[float], offset: int) -> List[Section]:
    b = _levels(np.asarray(bass_db, float), 18, 2)
    n = len(b)
    bounds = sorted(set([0] + list(range(offset, n, 8)) + [n]))
    blocks = [(a, z) for a, z in zip(bounds[:-1], bounds[1:]) if z > a]
    means = [b[a:z].mean() for a, z in blocks]
    loud = [m >= 6.0 for m in means]
    first = loud.index(True) if True in loud else len(blocks)
    last = len(loud) - 1 - loud[::-1].index(True) if True in loud else -1
    kinds = []
    for i, (m, l) in enumerate(zip(means, loud)):
        if l:
            k = "drop"
        elif i < first:
            k = "build" if i == first - 1 and m >= 3.5 else "intro"
        elif i > last:
            k = "outro"
        elif m < 3.5:
            k = "break"
        else:
            k = "build" if i + 1 < len(loud) and loud[i + 1] else "verse"
        kinds.append(k)
    out: List[Section] = []
    for (a, z), k in zip(blocks, kinds):
        if out and out[-1].kind == k:
            out[-1].end_bar = z
        else:
            out.append(Section(a, z, k))
    return out


def _energy(bass_db, rms_db, bpm) -> float:
    b = _levels(np.asarray(bass_db, float), 18, 2)
    dens = float((b >= 6).mean())
    loud = float(np.percentile(rms_db, 75)) if len(rms_db) else -12.0
    e = 1 + 9 * np.clip(0.45 * dens + 0.35 * (bpm - 120) / 20 + 0.2 * (loud + 16) / 10, 0, 1)
    return round(float(e), 1)


@lru_cache(maxsize=1)
def _raw():
    bars = json.loads((REF / "bars.json").read_text())
    grid = json.loads((REF / "grid2.json").read_text())
    vox = json.loads((REF / "vox.json").read_text())
    audio = json.loads(AUDIO.read_text()) if AUDIO.exists() else {}
    return bars, grid, vox, audio


def load_reference_tracks(keys: Optional[List[str]] = None) -> List[TrackAnalysis]:
    """TrackAnalysis for the reference tracks, in v2 order by default."""
    bars, grid, vox, audio = _raw()
    out = []
    for k in keys or V2_ORDER:
        title, artist, cam = META[k]
        B, G = bars[k], grid[k]
        feat = np.asarray(B["feat"], float)
        nb = len(feat)
        if k in audio:
            voc = audio[k]["vocal"][:nb]
            rms = audio[k]["rms"][:nb]
        else:   # digits → dB (inverse of vox.py's mapping, relative to 0 dB)
            voc = [-20 + 2.2 * int(c) for c in vox[k][:nb]]
            rms = [float(10 * np.log10((10 ** (r / 10)).sum())) for r in feat]
        bass = feat[:, 0].tolist()
        off = phrase_offset_heuristic(bass, voc)
        out.append(TrackAnalysis(
            path=f"reference/{k}", file_hash=f"ref-{k}", title=title, artist=artist,
            duration=float(G["T"]), sample_rate=44100, bpm=float(G["bpm"]),
            constant_tempo=(k != "09"), beats=list(B["beats"]), first_downbeat_index=int(B["first"]),
            key=cam, key_confidence=0.9, phrase_offset=off,
            bar_bass=bass, bar_mid=feat[:, 1].tolist(), bar_high=feat[:, 2].tolist(),
            bar_vocal=list(voc), bar_rms=list(rms),
            sections=sections_heuristic(bass, voc, off),
            energy=_energy(bass, rms, float(G["bpm"])),
        ))
    return out


def reference_plan() -> dict:
    return json.loads((REF / "plan.json").read_text())
