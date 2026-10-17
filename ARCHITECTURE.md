# PartyMix – architecture & contracts

PartyMix turns a folder of 10–30 songs into one continuous, beat-matched party mix,
exactly the way the hand-made "Half uur mix v2" was produced (see `reference/`).
Windows desktop app (PySide6), built with PyInstaller on GitHub Actions.
All code is Python 3.11 (must also run on 3.13 for local tests). UI strings are **Dutch**.

## Pipeline

```
files ──► analysis (per track, cached) ──► planner (order + cues) ──► render (stretch + mix + master) ──► WAV/FLAC/MP3
```

### 1. Analysis  (`partymix/analysis/`)  → `TrackAnalysis`
Per input file, cached in `%LOCALAPPDATA%/PartyMix/cache/<sha1-of-file>/`:
* decode with bundled ffmpeg to 44.1 kHz stereo float32 (`partymix/audio_io.py`)
* **beats/downbeats**: madmom `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100, min_bpm=100, max_bpm=160)`
  processed in ≤4-minute chunks (full tracks are fine; 30-min files OOM). Then fit a constant grid exactly like
  `reference/grid2.py` (local index via cumsum(round(diff/per0)), iterative robust polyfit, residual < 30 ms).
  If < 85 % inliers → tempo not constant → keep the (lightly smoothed) madmom beat list instead (see La Mamá case in `reference/bars.py`).
  `beats` = full list of beat times (s); `first_downbeat_index` = index into `beats` of the first downbeat.
* **key**: madmom `CNNKeyRecognitionProcessor` → Camelot code ("8A", "10B" …).
* **vocals**: ONNX model `UVR-MDX-NET-Inst_HQ_3.onnx` (MDX-Net; n_fft 6144, dim_f 3072, dim_t 2**8, hop 1024,
  compensate 1.022, primary stem = Instrumental; vocals = mix − instrumental) run with onnxruntime CPU, **no torch**.
  Writes `inst.flac` and `voc.flac` (float, lossless) in the cache dir.
* **per-bar features** (one row per bar from first downbeat): bass dB (20–150 Hz), mid dB (150–3 k), high dB (3–11 k),
  vocal dB (from vocal stem), RMS dB. See `reference/bars.py`, `reference/vox.py`.
* **phrase offset**: bar index (0–7) where 8-bar phrases start (novelty peaks on bass/vocal changes).
* **sections**: list of `Section(start_bar, end_bar, kind)` with kind ∈ intro, verse, build, drop, break, outro
  (bass-on & loud = drop/chorus, bass-off = break/intro/outro, rising = build).
* **energy**: 1–10 score (loudness of loud part, bass density, tempo, onset density).

### 2. Planner  (`partymix/planner/`)  → `MixPlan`
* `order.py`: choose the running order. Score per adjacent pair: tempo distance (prefer ≤ 3 %, never > 8 % unless a hard cut),
  Camelot compatibility (same, ±1, relative A/B = good), energy flow towards a target curve
  (warm-up → build → peak around 70–85 % → short finale), avoid same artist back-to-back. Solve with beam search / simulated annealing.
  User can pin first/last track or reorder manually (GUI).
* `cues.py`: for every pair pick the **segment** of each track (`start_bar`, `end_bar`) and the transition
  (`overlap_bars` 8/16, `preset`, options) so that:
  - every in/out point is on a phrase boundary;
  - incoming track's window ends where its first main section (drop/chorus with bass) starts;
  - swap point (bass + vocal hand-over) falls on the incoming track's bass entry;
  - no vocal-over-vocal (handled at render by the vocal hand-over, but prefer instrumental intros/outros);
  - total mix length ≈ target duration (user setting, default 30 min) → choose play lengths with DP.
  Reference: the hand-made plan in `reference/plan.json` + reasoning in `reference/PLAN_NOTES.md`.
* tempo curve: transition tempo per pair; track solo parts ramp linearly between transitions (see `reference/engine.py`).

### 3. Render  (`partymix/render/`)
* `stretch.py`: rubberband CLI (`-3` R3 engine) with a **per-beat timemap** (`-M`), for inst and voc stems separately.
* `mixer.py`: port of `reference/mix2.py`: per-track loudness match (−14 LUFS on loud part), envelopes for volume, bass (LR split @160 Hz,
  zero-phase), time-varying LP/HP filter, echo send, **vocal gain** (hand-over at swap point), presets
  `xfade, swap, swap_hp, editor, echo_out, echo_cut, double`.
* `master.py`: integrated loudness to −9.5 LUFS + true-peak limiter (−1 dBTP), output **WAV 24-bit / FLAC 24-bit** (lossless) and optional MP3 320
  (ffmpeg). Internal processing is float32/float64 throughout, 44.1 kHz (or 48 kHz if all sources are 48 kHz).
* `verify.py`: optional self-check (madmom over the result in chunks, beats-on-grid %) — like `reference/verify.py`.

### 4. GUI  (`partymix/gui/`)  PySide6
Drag & drop list → "Analyseren" (progress per track) → table (titel, BPM, key, energie, duur) + energy curve of the planned order →
settings (doelduur, uitvoer formaat, map) → "Mix maken" (progress, cancel) → open folder / play.
Long work runs in a worker process (multiprocessing), UI never blocks.

### 5. Packaging  (`packaging/`, `.github/workflows/`)
GitHub Actions `windows-latest`, Python 3.11: build madmom from git, PyInstaller onedir, bundle `ffmpeg.exe`, `rubberband.exe`,
the ONNX model; Inno Setup installer `PartyMix-Setup-x.y.z.exe`; attach to GitHub Release on tag `v*`.

## Shared code
* `partymix/model.py` – dataclasses below (single source of truth, JSON (de)serialisable).
* `partymix/paths.py` – locations of ffmpeg/rubberband/model (bundled dir when frozen, PATH otherwise), cache dir.
* `partymix/cli.py` – `python -m partymix analyze|plan|render|all <files...> -o out.flac --minutes 30`.

## Ownership (parallel agents – only edit your own files; ask the lead for contract changes)
| Area | Files |
|---|---|
| analysis | `partymix/analysis/**`, `partymix/audio_io.py`, `tests/test_analysis*.py` |
| planner | `partymix/planner/**`, `tests/test_planner*.py` |
| render | `partymix/render/**`, `tests/test_render*.py` |
| gui + cli | `partymix/gui/**`, `partymix/cli.py`, `partymix/__main__.py` |
| packaging | `packaging/**`, `.github/**`, `README.md`, `pyproject.toml`, `requirements*.txt` |
| lead | `partymix/model.py`, `partymix/paths.py`, `ARCHITECTURE.md`, `reference/**` |
