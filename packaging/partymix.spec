# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PartyMix (onedir).

Build (from the repo root, after `python packaging/fetch_tools.py`):
    pyinstaller packaging/partymix.spec --noconfirm --distpath dist --workpath build/pyinstaller

Output: dist/PartyMix/
    PartyMix.exe        windowed GUI (no console)
    partymix-cli.exe    same program with a console: CLI sub-commands, `--selftest`, `--bundle-selftest`
    _internal/          Python runtime, libraries, data  (== sys._MEIPASS == paths.bundle_dir())
        bin/            ffmpeg.exe, ffprobe.exe, rubberband.exe, rubberband-r3.exe, sndfile.dll
        models/         UVR-MDX-NET-Inst_HQ_3.onnx
        madmom/models/  madmom's pretrained networks (loaded from the package directory)
        licenses/       third-party licence texts

Environment overrides (used for the local Linux dry run):
    PARTYMIX_ENTRY      alternative entry script instead of packaging/launcher.py
    PARTYMIX_TOOLS_DIR  where bin/, models/, licenses/ come from (default: packaging/)
"""
import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

SPEC_DIR = Path(SPECPATH).resolve()          # noqa: F821  (SPECPATH is injected by PyInstaller)
ROOT = SPEC_DIR.parent
TOOLS = Path(os.environ.get("PARTYMIX_TOOLS_DIR", SPEC_DIR)).resolve()
ENTRY = Path(os.environ.get("PARTYMIX_ENTRY", SPEC_DIR / "launcher.py")).resolve()
IS_WIN = sys.platform == "win32"
# make `partymix` and `partymix_selftest` importable for the collect_* helpers below
for _p in (str(SPEC_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

version = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "partymix" / "__init__.py").read_text()).group(1)
print(f"[partymix.spec] version {version}, entry {ENTRY}, tools {TOOLS}")


def _optional(fn, *args, **kw):
    """Hooks helpers raise/return nothing when a package is missing; keep the build going."""
    try:
        return fn(*args, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[partymix.spec] WARNING {fn.__name__}{args}: {exc}")
        return []


# ---------------------------------------------------------------- data files
datas = []
binaries = []
hiddenimports = []

# Our own package: every submodule (the launcher uses runpy, so static analysis sees nothing)
# plus non-Python files (e.g. partymix/gui/assets/*).
hiddenimports += collect_submodules("partymix")
hiddenimports += ["partymix_selftest"]  # packaging/partymix_selftest.py, used by `--bundle-selftest`
datas += collect_data_files("partymix", excludes=["**/__pycache__/**"])

# madmom: Cython extensions + the pretrained models that it loads from <package>/models/,
# and its dist-info (madmom/__init__.py calls importlib.metadata.distribution("madmom")).
hiddenimports += collect_submodules("madmom", filter=lambda n: ".tests" not in n and not n.endswith("conftest"))
datas += collect_data_files("madmom", excludes=["**/tests/**", "**/__pycache__/**"])
datas += copy_metadata("madmom")
datas += _optional(copy_metadata, "mido")          # mido/version.py reads its metadata
# madmom's model files are pickles that reference numpy 1.x paths (numpy.core.multiarray._reconstruct);
# numpy 2 serves them from lazily-imported shim modules that static analysis does not see.
hiddenimports += collect_submodules("numpy.core") + collect_submodules("numpy._core")
hiddenimports += ["mido", "scipy.signal", "scipy.ndimage", "scipy.interpolate", "scipy.spatial"]

# onnxruntime (CPU): native DLLs/so's + capi; metadata is queried at import.
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += collect_submodules("onnxruntime.capi")
datas += _optional(copy_metadata, "onnxruntime")

# soundfile ships libsndfile in the _soundfile_data package (hooks-contrib also handles this).
binaries += _optional(collect_dynamic_libs, "_soundfile_data")
datas += _optional(collect_data_files, "_soundfile_data")
hiddenimports += ["soundfile", "_soundfile", "pyloudnorm"]

# Bundled tools / model / licences (fetched by packaging/fetch_tools.py). Added as data so
# PyInstaller does not try to rewrite or dependency-scan the third-party executables.
missing = []
for sub, required in (("bin", ["ffmpeg", "ffprobe", "rubberband"]), ("models", []), ("licenses", [])):
    src = TOOLS / sub
    if not src.is_dir():
        missing.append(sub)
        continue
    for f in sorted(src.rglob("*")):
        if f.is_file() and not f.name.endswith(".part"):
            datas.append((str(f), str(Path(sub) / f.relative_to(src).parent)))
    for tool in required:
        exts = (".exe",) if IS_WIN else (".exe", "")
        if not any((src / (tool + ext)).is_file() for ext in exts):
            missing.append(f"{sub}/{tool}")
if missing:
    msg = f"[partymix.spec] missing bundled files: {missing} - run `python packaging/fetch_tools.py`"
    if IS_WIN and not os.environ.get("PARTYMIX_ALLOW_MISSING_TOOLS"):
        raise SystemExit(msg)
    print(msg + " (continuing: non-Windows dry run)")

for doc in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md"):
    if (ROOT / doc).exists():
        datas.append((str(ROOT / doc), "."))

# ---------------------------------------------------------------- analysis
EXCLUDES = [
    # never wanted (and some would add hundreds of MB if present in the build env)
    "torch", "torchaudio", "torchvision", "tensorflow", "onnx", "onnx2torch",
    "matplotlib", "IPython", "jupyter", "notebook", "pandas",
    "tkinter", "_tkinter", "PyQt5", "PyQt6", "PySide2",
    # build/test tooling that leaks in through madmom/scipy/cffi optional imports
    "Cython", "cython", "pyximport", "pytest", "_pytest", "pluggy", "pytestqt",
    "setuptools", "pkg_resources", "wheel", "pip",
    # unused heavy Qt modules
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtQuick3D", "PySide6.QtPdf",
    "PySide6.QtDesigner", "PySide6.QtBluetooth", "PySide6.QtLocation", "PySide6.QtSensors",
]

a = Analysis(  # noqa: F821
    [str(ENTRY)],
    pathex=[str(ROOT), str(SPEC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,  # keep docstrings/asserts: madmom/scipy use them
)
pyz = PYZ(a.pure)  # noqa: F821

icon = ROOT / "partymix" / "gui" / "assets" / "icon.ico"
icon_arg = [str(icon)] if icon.exists() else None
if icon_arg is None:
    print("[partymix.spec] no partymix/gui/assets/icon.ico - using the default icon")

common = dict(
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks Qt/onnxruntime DLLs and triggers antivirus false positives
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)

gui_exe = EXE(pyz, a.scripts, [], name="PartyMix", console=False, **common)  # noqa: F821
cli_exe = EXE(pyz, a.scripts, [], name="partymix-cli", console=True, **common)  # noqa: F821

coll = COLLECT(  # noqa: F821
    gui_exe,
    cli_exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PartyMix",
)
