"""Where bundled tools/models live and where the cache goes. Owned by the lead."""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path

APP_NAME = "PartyMix"
MODEL_NAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/" + MODEL_NAME


def frozen() -> bool:
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Directory with bundled binaries/models (PyInstaller) or the repo root in development."""
    if frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _tool(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    for cand in (bundle_dir() / "bin" / exe, bundle_dir() / exe):
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} niet gevonden (verwacht in {bundle_dir() / 'bin'} of op PATH)")
    return found


def ffmpeg() -> str:
    return _tool("ffmpeg")


def ffprobe() -> str:
    return _tool("ffprobe")


def rubberband() -> str:
    return _tool("rubberband")


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = Path(os.environ.get("PARTYMIX_CACHE", data_dir() / "cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path() -> Path:
    """Bundled model, else downloaded copy in the data dir (download handled by analysis.vocals)."""
    for cand in (bundle_dir() / "models" / MODEL_NAME, data_dir() / "models" / MODEL_NAME):
        if cand.exists():
            return cand
    return data_dir() / "models" / MODEL_NAME
