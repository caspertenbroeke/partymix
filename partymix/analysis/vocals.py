"""Vocal / instrumental separation with the MDX-Net model ``UVR-MDX-NET-Inst_HQ_3.onnx``.

Pure numpy + onnxruntime port of audio-separator 0.47's ``MDXSeparator`` (default settings:
segment_size 256, overlap 0.25, hop 1024, batch 1, no denoise) and its torch STFT helper,
so no torch is needed at runtime.

Model parameters: n_fft 6144, dim_f 3072, dim_t 2**8 = 256 frames, hop 1024,
compensate 1.022, primary stem = Instrumental.

Differences to audio-separator (deliberate, documented):
* the input is peak-normalised to 0.9 like audio-separator, but the output is scaled back by
  the *same* factor (audio-separator multiplies by the peak instead, which makes
  ``vocals = mix − 1.022·inst`` inconsistent for files that peak above 0.9);
* the instrumental is returned already multiplied by ``compensate`` so that
  ``inst + vocals == mix`` exactly (audio-separator writes the uncompensated instrumental);
* no peak normalisation of the written stems.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .. import paths

N_FFT = 6144
HOP = 1024
DIM_F = 3072
DIM_T = 256
COMPENSATE = 1.022
OVERLAP = 0.25
NORM_PEAK = 0.9
MODEL_SIZE = 66759214
MODEL_TAIL_MD5 = "55657dd70583b0fedfba5f67df11d711"   # UVR hash: md5 of the last 10 000 KiB

ProgressCb = Optional[Callable[[float], None]]
CancelCb = Optional[Callable[[], bool]]

__all__ = ["separate", "ensure_model", "MdxSeparator"]


# ----------------------------------------------------------------------------- model file
def _tail_md5(p: Path) -> str:
    with open(p, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


def ensure_model(progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Return the path of the ONNX model, downloading it from ``paths.MODEL_URL`` if missing.

    ``progress(fraction, message_nl)`` is called during the download.
    """
    p = paths.model_path()
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    req = urllib.request.Request(paths.MODEL_URL, headers={"User-Agent": "PartyMix"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or MODEL_SIZE)
        done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if progress:
                progress(min(done / total, 1.0),
                         f"Stemscheidingsmodel downloaden… {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    if _tail_md5(tmp) != MODEL_TAIL_MD5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Download van het stemscheidingsmodel is beschadigd (controlesom klopt niet)")
    os.replace(tmp, p)
    return p


# ----------------------------------------------------------------------------- STFT (torch-compatible)
_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic hann


def _stft(x: np.ndarray) -> np.ndarray:
    """``torch.stft(center=True, pad_mode='reflect', hann periodic)`` for ``x [C, T]`` → ``[C, F, frames]``."""
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + (xp.shape[1] - N_FFT) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(xp, N_FFT, axis=1)[:, ::HOP][:, :n_frames]
    return np.fft.rfft(fr * _WIN, axis=-1).transpose(0, 2, 1)          # [C, F, frames] complex


def _istft(S: np.ndarray, length: int) -> np.ndarray:
    """``torch.istft(center=True)`` for ``S [C, F, frames]`` → ``[C, length]``."""
    C, _, n_frames = S.shape
    fr = np.fft.irfft(S.transpose(0, 2, 1), n=N_FFT, axis=-1).astype(np.float32) * _WIN
    total = N_FFT + HOP * (n_frames - 1)
    y = np.zeros((C, total), np.float32)
    env = np.zeros(total, np.float32)
    w2 = _WIN ** 2
    for t in range(n_frames):
        y[:, t * HOP:t * HOP + N_FFT] += fr[:, t]
        env[t * HOP:t * HOP + N_FFT] += w2
    pad = N_FFT // 2
    y, env = y[:, pad:pad + length], env[pad:pad + length]
    return y / np.where(env > 1e-11, env, 1.0)


# ----------------------------------------------------------------------------- separator
class MdxSeparator:
    """Holds the onnxruntime session; reuse it for several tracks."""

    def __init__(self, model: Optional[Path] = None, threads: Optional[int] = None,
                 device: Optional[str] = None):
        import onnxruntime as ort
        model = Path(model) if model else ensure_model()
        self.chunk = HOP * (DIM_T - 1)          # 261120 samples ≈ 5.9 s
        self.trim = N_FFT // 2
        want = (device or os.environ.get("PARTYMIX_DEVICE", "auto")).lower()
        avail = set(ort.get_available_providers())
        gpu_order = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
        candidates = ([] if want == "cpu" else gpu_order) + ["CPUExecutionProvider"]
        self.session = None
        for prov in candidates:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            so.enable_mem_pattern = False
            if prov == "CPUExecutionProvider":
                so.intra_op_num_threads = threads or max(1, os.cpu_count() or 1)
                so.inter_op_num_threads = 1
                # the arena keeps ~1.2 GB of peak activations alive between runs; without it the
                # process stays at ~1.4 GB with no measurable speed loss
                so.enable_cpu_mem_arena = False
            else:
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                sess = ort.InferenceSession(str(model), sess_options=so, providers=[prov])
                if prov != "CPUExecutionProvider":
                    # make sure the GPU really works (drivers!) and is really fast: without a real
                    # video card DirectML runs on the software "Basic Render" adapter, slower than the CPU
                    import time
                    z = np.zeros((1, 4, DIM_F, DIM_T), np.float32)
                    sess.run(None, {"input": z})
                    t0 = time.perf_counter()
                    sess.run(None, {"input": z})
                    if time.perf_counter() - t0 > float(os.environ.get("PARTYMIX_GPU_MAX_S", "1.0")):
                        continue
                self.session, self.provider = sess, prov
                break
            except Exception:
                continue
        if self.session is None:
            raise RuntimeError("onnxruntime kon het stemscheidingsmodel niet laden")
        self.on_gpu = self.provider != "CPUExecutionProvider"
        self.device_name = {"CUDAExecutionProvider": "videokaart (CUDA)",
                            "DmlExecutionProvider": "videokaart (DirectML)"}.get(self.provider, "processor")
        # several chunks per model call on the GPU; on the CPU batching gives no speed-up
        self.batch = int(os.environ.get("PARTYMIX_BATCH", "4" if self.on_gpu else "1"))

    def _spec(self, wave: np.ndarray) -> np.ndarray:
        S = _stft(wave)[:, :DIM_F]                                        # [2, 3072, 256]
        spek = np.stack([S.real, S.imag], axis=1).reshape(4, DIM_F, -1).astype(np.float32)
        spek[:, :3, :] = 0.0
        return spek

    def _run_batch(self, waves: list) -> list:
        """Model on chunks ``[2, chunk]`` → instrumental chunks ``[2, chunk]`` (one model call)."""
        spek = np.stack([self._spec(w) for w in waves])                  # [B, 4, 3072, 256]
        outs = self.session.run(None, {"input": spek})[0]                # [B, 4, 3072, 256]
        res = []
        for out in outs:
            out = out.reshape(2, 2, DIM_F, -1)
            full = np.zeros((2, N_FFT // 2 + 1, out.shape[-1]), np.complex64)
            full[:, :DIM_F] = out[:, 0] + 1j * out[:, 1]
            res.append(_istft(full, self.chunk))
        return res

    def _run(self, wave: np.ndarray) -> np.ndarray:
        return self._run_batch([wave])[0]

    def instrumental(self, mix: np.ndarray, overlap: float = OVERLAP, progress: ProgressCb = None,
                     cancel: CancelCb = None) -> np.ndarray:
        """Raw (uncompensated) instrumental estimate for ``mix [2, N]`` (already normalised)."""
        n = mix.shape[1]
        chunk, trim = self.chunk, self.trim
        gen = chunk - 2 * trim
        pad = gen + trim - (n % gen)
        mixture = np.concatenate([np.zeros((2, trim), np.float32), mix.astype(np.float32),
                                  np.zeros((2, pad), np.float32)], axis=1)
        L = mixture.shape[1]
        step = int((1 - overlap) * chunk)
        result = np.zeros((2, L), np.float32)
        divider = np.zeros(L, np.float32)
        starts = list(range(0, L, step))
        B = max(1, self.batch)
        for k0 in range(0, len(starts), B):
            if cancel is not None and cancel():
                raise InterruptedError("geannuleerd")
            group = starts[k0:k0 + B]
            parts = []
            for i in group:
                end = min(i + chunk, L)
                part = mixture[:, i:end]
                if end - i < chunk:
                    part = np.concatenate([part, np.zeros((2, chunk - (end - i)), np.float32)], axis=1)
                parts.append(part)
            try:
                ys = self._run_batch(parts)
            except Exception:
                if B == 1:
                    raise
                ys = [self._run(pt) for pt in parts]   # e.g. GPU out of memory: one at a time
                self.batch = B = 1
            for i, y in zip(group, ys):
                end = min(i + chunk, L)
                y = y[:, :end - i]
                if overlap != 0:
                    w = np.hanning(end - i).astype(np.float32)
                    result[:, i:end] += y * w
                    divider[i:end] += w
                else:
                    result[:, i:end] += y
                    divider[i:end] += 1
            if progress:
                progress(min(1.0, (k0 + len(group)) / len(starts)))
        with np.errstate(invalid="ignore", divide="ignore"):
            out = result / divider
        return out[:, trim:L - trim][:, :n]

    def separate(self, audio: np.ndarray, progress: ProgressCb = None, cancel: CancelCb = None,
                 overlap: float = OVERLAP) -> Tuple[np.ndarray, np.ndarray]:
        """Split ``audio [n, 2]`` into ``(instrumental, vocals)``, both ``[n, 2]`` float32,
        with ``instrumental + vocals == audio``."""
        x = np.asarray(audio, np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        mix = x.T
        peak = float(np.abs(mix).max()) if mix.size else 0.0
        if peak == 0.0 or not np.isfinite(peak):
            z = np.zeros_like(x)
            return x.copy(), z
        scale = NORM_PEAK / peak if peak > NORM_PEAK else 1.0
        inst = self.instrumental(mix * scale, overlap, progress, cancel) * (COMPENSATE / scale)
        inst = np.nan_to_num(inst).T.astype(np.float32)
        return np.ascontiguousarray(inst), np.ascontiguousarray(x - inst)


_SEP: Optional[MdxSeparator] = None


def get_separator(model: Optional[Path] = None) -> MdxSeparator:
    """Per-process cached :class:`MdxSeparator` (GPU when available, else CPU)."""
    global _SEP
    if _SEP is None:
        _SEP = MdxSeparator(model)
    return _SEP


def device_name() -> str:
    """Dutch name of the device used for separation ("videokaart (DirectML)" / "processor")."""
    return get_separator().device_name


def separate(audio: np.ndarray, progress: ProgressCb = None, cancel: CancelCb = None,
             model: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper with a per-process cached :class:`MdxSeparator`."""
    return get_separator(model).separate(audio, progress=progress, cancel=cancel)


if __name__ == "__main__":  # quick manual check: python -m partymix.analysis.vocals in.wav
    import sys
    from ..audio_io import decode, write_flac
    a, sr = decode(sys.argv[1])
    t0 = time.time()
    inst, voc = separate(a, progress=lambda f: print(f"\r{f:5.1%}", end="", flush=True))
    dt = time.time() - t0
    print(f"\n{len(a) / sr:.1f} s audio in {dt:.1f} s ({len(a) / sr / dt:.2f}x realtime)")
    write_flac("inst.flac", inst, sr)
    write_flac("voc.flac", voc, sr)
