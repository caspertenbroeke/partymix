"""Rendering: stretch (rubberband) -> mix (port of reference/mix2.py) -> master -> files.

Main entry point::

    render_mix(plan, analyses, settings, progress_cb, cancel_event) -> list[str]

``progress_cb(fraction 0..1, message_nl)``; ``cancel_event`` is anything with ``is_set()``
(threading.Event / multiprocessing.Event).  Raises :class:`RenderCancelled` on cancel.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .. import paths
from ..model import MixPlan, RenderSettings, TrackAnalysis
from .common import RenderCancelled, RenderError, check_cancel, sub_progress
from .layout import Layout, compute_layout, estimate_duration as _estimate_core
from .master import CueEntry, MasterReport, master_to_files
from .mixer import OUTPUT_TAIL_SEC, MasterBuffer, mix_track, output_frames, PRESETS
from .stretch import iter_stretched

__all__ = ["render_mix", "estimate_duration", "compute_layout", "Layout", "RenderCancelled",
           "RenderError", "PRESETS", "track_cues", "order_analyses", "last_report"]

last_report: Dict = {}


def estimate_duration(plan: MixPlan, analyses: Optional[Sequence[TrackAnalysis]] = None) -> float:
    """Length in seconds of the file :func:`render_mix` will write for ``plan``."""
    return _estimate_core(plan, analyses) + OUTPUT_TAIL_SEC


def order_analyses(plan: MixPlan, analyses: Sequence[TrackAnalysis]) -> List[TrackAnalysis]:
    """Analyses in plan order (matched by ``file_hash``; falls back to position)."""
    by_hash = {a.file_hash: a for a in analyses}
    out = []
    for i, t in enumerate(plan.tracks):
        a = by_hash.get(t.track_hash)
        if a is None:
            if len(analyses) == len(plan.tracks) and not by_hash.keys() & {x.track_hash for x in plan.tracks}:
                a = analyses[i]
            else:
                raise RenderError(f"Analyse ontbreekt voor nummer {i + 1} ({t.track_hash[:8]})")
        out.append(a)
    return out


def track_cues(plan: MixPlan, layout: Layout, analyses: Sequence[TrackAnalysis]) -> List[CueEntry]:
    """Chapter times: the moment each track takes over (its bass/vocal swap point)."""
    cues = []
    for i, a in enumerate(analyses):
        x = layout.offsets[i]
        if i > 0:
            tr = plan.transitions[i - 1]
            sw = float(tr.options.get("swap", 0.5))
            x += max(0.0, min(1.0, sw)) * tr.overlap_bars
        cues.append(CueEntry(time=layout.t(x), title=a.title or Path(a.path).stem, artist=a.artist or ""))
    return cues


def _tmp_root() -> str:
    d = paths.data_dir() / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def render_mix(plan: MixPlan, analyses: Sequence[TrackAnalysis], settings: RenderSettings,
               progress_cb: Optional[Callable[[float, str], None]] = None, cancel_event=None, *,
               max_jobs: int = 2, workdir: Optional[str] = None, premaster_path: Optional[str] = None,
               title: Optional[str] = None) -> List[str]:
    """Render ``plan`` to the files requested in ``settings``. Returns the written paths
    (audio files, ``.cue``, ``.txt``, and ``*_controle.json`` when ``settings.verify``).

    ``premaster_path``: also save the un-mastered float mix as 32-bit float WAV (debug/tests).
    """
    global last_report
    t_start = time.time()
    prog = progress_cb or (lambda f, m: None)
    an = order_analyses(plan, analyses)
    rates = {a.sample_rate for a in an}
    if len(rates) != 1:
        raise RenderError(f"Nummers hebben verschillende samplerates: {sorted(rates)}")
    sr = rates.pop()
    layout = compute_layout(plan, an)
    n = layout.n
    own_tmp = workdir is None
    work = workdir or tempfile.mkdtemp(prefix="render_", dir=_tmp_root())
    os.makedirs(work, exist_ok=True)
    verify_share = 0.15 if settings.verify else 0.0
    p_mix = sub_progress(prog, 0.0, 0.75 - verify_share * 0.5)
    p_master = sub_progress(prog, 0.75 - verify_share * 0.5, 1.0 - verify_share)
    out_files: List[str] = []
    try:
        frames = output_frames(layout, sr)
        master = MasterBuffer(frames, os.path.join(work, "premaster.f32"))
        p_mix(0.0, f"Tempo aanpassen: {an[0].title} (1/{n})")
        for res in iter_stretched(plan, an, layout, os.path.join(work, "stretch"), sr=sr,
                                  max_jobs=max_jobs, lookahead=1, cancel_event=cancel_event):
            i = res.index
            check_cancel(cancel_event)
            p_mix((i + 0.5) / n, f"Mixen: {an[i].title} ({i + 1}/{n})")
            mix_track(i, res, plan, layout, master, sr)
            res.cleanup()
            nxt = f"Tempo aanpassen: {an[i + 1].title} ({i + 2}/{n})" if i + 1 < n else "Mixen klaar"
            p_mix((i + 1) / n, nxt)
        if premaster_path:
            import soundfile as sf
            with sf.SoundFile(premaster_path, "w", samplerate=sr, channels=2, subtype="FLOAT") as fo:
                for _, x in master.chunks(sr * 30):
                    fo.write(x)
        cues = track_cues(plan, layout, an)
        rep: MasterReport = master_to_files(
            master.read, frames, sr, settings.out_path, settings.formats, work, cues,
            target_lufs=settings.target_lufs, true_peak_db_=settings.true_peak_db,
            bit_depth=settings.bit_depth, title=title or Path(settings.out_path).stem,
            progress=p_master, cancel_event=cancel_event)
        master.remove()
        out_files = list(rep.files)
        last_report = {"input_lufs": rep.input_lufs, "gain_db": rep.gain_db, "output_lufs": rep.output_lufs,
                       "true_peak_db": rep.output_true_peak_db, "max_gain_reduction_db": rep.max_gain_reduction_db,
                       "duration_sec": frames / sr, "render_sec": None}
        if settings.verify:
            from .verify import verify_mix
            audio = next((f for f in out_files if f.lower().endswith((".flac", ".wav"))), out_files[0])
            vr = verify_mix(audio, layout, [a.title for a in an],
                            sub_progress(prog, 1.0 - verify_share, 1.0), cancel_event)
            jp = str(Path(settings.out_path).with_suffix("")) + "_controle.json"
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(vr, f, indent=1, ensure_ascii=False)
            out_files.append(jp)
            last_report["verify"] = vr
        last_report["render_sec"] = time.time() - t_start
        prog(1.0, "Klaar")
        return out_files
    finally:
        if own_tmp:
            shutil.rmtree(work, ignore_errors=True)
