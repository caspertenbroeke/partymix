# Third-party notices

PartyMix itself is licensed under the **GNU General Public License v3.0 or later** (see `LICENSE`).
The Windows build bundles the components below. Their licence texts are shipped in the app folder
(`_internal/licenses/`, `_internal/*.dist-info/`, `_internal/madmom/models/LICENSE`).

## Summary — what this means for you

* **Non-commercial use only.** The beat, downbeat and key detection uses madmom's pretrained models,
  which are licensed **CC BY-NC-SA 4.0**. You may not use PartyMix (as distributed, with these models)
  for commercial purposes. A paid DJ gig, or a mix made for a business, counts as commercial use.
* **GPL.** The app ships Rubber Band and an FFmpeg GPL build. That is why the whole distribution is
  GPL-3.0-or-later. You get the source code at https://github.com/caspertenbroeke/partymix, and the
  sources of the bundled tools are listed below.
* PartyMix does not give you any rights to the **music** you mix. You are responsible for it.

> Note on combining licences: the CC BY-NC-SA model files are separate data files that are only loaded
> at run time. They are distributed *alongside* the GPL program (an "aggregate" in GPL terms) and are
> not part of the GPL-licensed work. The non-commercial restriction applies to the models. If you need
> commercial use, replace the madmom models with models whose licence allows it.

## Bundled components

| Component | Version | Licence | Use in PartyMix | Source |
|---|---|---|---|---|
| **madmom** (code) | 0.17.dev0, git `27f032e` | BSD-2-Clause | beat/downbeat tracking, key detection | https://github.com/CPJKU/madmom |
| **madmom models** | `madmom_models` submodule | **CC BY-NC-SA 4.0** | pretrained RNN/CNN weights | https://github.com/CPJKU/madmom_models |
| **Rubber Band** (CLI `rubberband.exe`, `rubberband-r3.exe`) | 4.0.0 | GPL-2.0-or-later | time-stretching (tempo matching) | https://breakfastquay.com/rubberband/ · https://github.com/breakfastquay/rubberband |
| libsndfile (`sndfile.dll`, comes with Rubber Band) | – | LGPL-2.1-or-later | audio file I/O for rubberband | https://github.com/libsndfile/libsndfile |
| **FFmpeg** (`ffmpeg.exe`, `ffprobe.exe`), gyan.dev "essentials" build | 9.0.2 | GPL-3.0 (built with `--enable-gpl --enable-version3`) | decoding input files, MP3 export | https://ffmpeg.org · build: https://www.gyan.dev/ffmpeg/builds/ · https://github.com/GyanD/codexffmpeg/releases/tag/9.0.2 · exact source commit in `_internal/licenses/ffmpeg/README.txt` |
| **UVR-MDX-NET-Inst_HQ_3** ONNX model | – | MIT (per the Ultimate Vocal Remover project; credit required) | vocal/instrumental separation | https://github.com/Anjok07/ultimatevocalremovergui · file: https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models |
| ONNX Runtime (CPU) | ≥ 1.20 | MIT | runs the UVR model | https://github.com/microsoft/onnxruntime |
| Qt 6 / PySide6 / shiboken6 | ≥ 6.7 | LGPL-3.0 (dynamically linked, replaceable DLLs in `_internal/PySide6/`) | user interface | https://www.qt.io · https://wiki.qt.io/Qt_for_Python |
| NumPy | 2.x | BSD-3-Clause (+ bundled OpenBLAS: BSD-3-Clause; gfortran runtime: GPL-3.0 with GCC Runtime Library Exception) | numerics | https://numpy.org |
| SciPy | ≥ 1.13 | BSD-3-Clause (+ OpenBLAS / gfortran runtime as above) | filters, signal processing | https://scipy.org |
| python-soundfile | ≥ 0.12 | BSD-3-Clause | reading/writing WAV/FLAC | https://github.com/bastibe/python-soundfile |
| libsndfile (`_soundfile_data`) | 1.2.x | LGPL-2.1-or-later | used by python-soundfile | https://github.com/libsndfile/libsndfile |
| pyloudnorm | ≥ 0.1.1 | MIT | loudness (LUFS) measurement | https://github.com/csteinmetz1/pyloudnorm |
| mido | ≥ 1.3 | MIT | madmom dependency | https://github.com/mido/mido |
| cffi / pycparser | – | MIT-0 / BSD-3-Clause | python-soundfile dependency | https://github.com/python-cffi/cffi |
| Python | 3.11 | PSF License | runtime | https://www.python.org |
| PyInstaller bootloader | 6.x | GPL-2.0 with bootloader exception | app launcher | https://pyinstaller.org |

## Attribution

* **madmom**: Sebastian Böck, Filip Korzeniowski, Jan Schlüter, Florian Krebs, Gerhard Widmer,
  "madmom: a new Python Audio and Music Signal Processing Library", Proc. ACM Multimedia 2016.
  Models © Department of Computational Perception, Johannes Kepler University Linz, and OFAI —
  licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). PartyMix uses
  the models unchanged.
* **Ultimate Vocal Remover**: the UVR-MDX-NET models were trained by the UVR core developers
  (Anjok07 and aufr33). The MDX-Net architecture comes from Kuielab & Woosung Choi.
  Thank you to UVR and its developers.
* **Rubber Band Library**: © Particular Programs Ltd (Breakfast Quay), Chris Cannam.
* **FFmpeg**: © the FFmpeg developers. Windows build by Gyan Doshi (gyan.dev).

## Written offer / source availability

PartyMix source: https://github.com/caspertenbroeke/partymix (every release is tagged).
The bundled GPL/LGPL binaries are unmodified upstream builds. You can download them from the links
above. `packaging/fetch_tools.py` lists the exact URLs and SHA-256 checksums used for each release.
If one of these links stops working, open an issue on the PartyMix repository and we will send you the
corresponding source code.
