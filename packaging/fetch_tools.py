#!/usr/bin/env python3
"""Download and verify the third-party tools/models that PartyMix bundles.

Result (layout expected by partymix/paths.py once frozen, see packaging/partymix.spec):

    packaging/bin/ffmpeg.exe, ffprobe.exe          (gyan.dev GPLv3 "essentials" static build)
    packaging/bin/rubberband.exe, rubberband-r3.exe, sndfile.dll   (Breakfast Quay, GPL)
    packaging/models/UVR-MDX-NET-Inst_HQ_3.onnx    (UVR, MIT, credit required)
    packaging/licenses/<component>/...             (licence texts shipped with the app)

Every download is pinned by URL *and* SHA-256. To upgrade a component, change the URL,
run `python packaging/fetch_tools.py --print-hashes` and paste the new hash.

Pure standard library so it runs before any dependency is installed.
Usage:  python packaging/fetch_tools.py [--dest packaging] [--cache DIR] [--print-hashes]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name, url, sha256, {member-suffix-in-zip: target-relative-path}  (None mapping = file is used as-is)
FFMPEG_VERSION = "9.0.2"
RUBBERBAND_VERSION = "4.0.0"
COMPONENTS = [
    (
        "ffmpeg",
        f"https://github.com/GyanD/codexffmpeg/releases/download/{FFMPEG_VERSION}/"
        f"ffmpeg-{FFMPEG_VERSION}-essentials_build.zip",
        "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
        {
            "/bin/ffmpeg.exe": "bin/ffmpeg.exe",
            "/bin/ffprobe.exe": "bin/ffprobe.exe",
            "/LICENSE": "licenses/ffmpeg/LICENSE.txt",
            "/README.txt": "licenses/ffmpeg/README.txt",
        },
    ),
    (
        "rubberband",
        f"https://breakfastquay.com/files/releases/rubberband-{RUBBERBAND_VERSION}-gpl-executable-windows.zip",
        "f2d47fc64dbb42f6cc62edf7933ac4fa89d8f0ef8b9cf97b6afc263a7fe05644",
        {
            "/rubberband.exe": "bin/rubberband.exe",
            "/rubberband-r3.exe": "bin/rubberband-r3.exe",
            "/sndfile.dll": "bin/sndfile.dll",  # required by rubberband.exe (only other import is KERNEL32)
            "/COPYING.txt": "licenses/rubberband/COPYING.txt",
            "/README.txt": "licenses/rubberband/README.txt",
        },
    ),
    (
        "uvr-model",
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
        "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc",
        None,
    ),
]
MODEL_TARGET = "models/UVR-MDX-NET-Inst_HQ_3.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"  download {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "partymix-build"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def fetch(url: str, expected: str, cache: Path, print_only: bool) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.rsplit("/", 1)[1]
    if path.exists() and sha256(path) == expected:
        print(f"  cached   {path.name}")
        return path
    download(url, path)
    got = sha256(path)
    if print_only:
        print(f"  sha256   {got}  {path.name}")
    elif got != expected:
        path.unlink()
        raise SystemExit(f"SHA-256 mismatch for {url}\n  expected {expected}\n  got      {got}")
    return path


def extract(zip_path: Path, mapping: dict[str, str], dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for suffix, target in mapping.items():
            hits = [n for n in names if n.endswith(suffix) and not n.endswith("/")]
            if len(hits) != 1:
                raise SystemExit(f"{zip_path.name}: expected exactly one member ending in {suffix}, got {hits}")
            out = dest / target
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(hits[0]) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print(f"  extract  {target}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=HERE, help="output root (default: packaging/)")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("PARTYMIX_DL_CACHE", Path(tempfile.gettempdir()) / "partymix-dl")))
    ap.add_argument("--print-hashes", action="store_true", help="only print hashes (for upgrading pins)")
    args = ap.parse_args(argv)

    for name, url, expected, mapping in COMPONENTS:
        print(f"[{name}]")
        path = fetch(url, expected, args.cache, args.print_hashes)
        if args.print_hashes:
            continue
        if mapping is None:
            out = args.dest / MODEL_TARGET
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            print(f"  copy     {MODEL_TARGET}")
        else:
            extract(path, mapping, args.dest)

    if not args.print_hashes:
        lic = args.dest / "licenses" / "uvr-model"
        lic.mkdir(parents=True, exist_ok=True)
        (lic / "NOTICE.txt").write_text(
            "UVR-MDX-NET-Inst_HQ_3.onnx\n"
            "Trained by the Ultimate Vocal Remover (UVR) core developers (Anjok07, aufr33).\n"
            "https://github.com/Anjok07/ultimatevocalremovergui - distributed under the MIT license;\n"
            "the UVR README asks third-party applications using the models to credit UVR and its developers.\n"
            "Model file: https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models\n",
            encoding="utf-8",
        )
        print("OK - tools ready in", args.dest.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
