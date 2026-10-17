"""Musical key detection (madmom ``CNNKeyRecognitionProcessor``) → Camelot code."""
from __future__ import annotations

from typing import Tuple

import numpy as np

CHUNK_S = 240.0

# madmom KEY_LABELS order -> Camelot
_CAMELOT = {
    "A major": "11B", "Bb major": "6B", "B major": "1B", "C major": "8B", "Db major": "3B",
    "D major": "10B", "Eb major": "5B", "E major": "12B", "F major": "7B", "F# major": "2B",
    "G major": "9B", "Ab major": "4B",
    "A minor": "8A", "Bb minor": "3A", "B minor": "10A", "C minor": "5A", "C# minor": "12A",
    "D minor": "7A", "D# minor": "2A", "E minor": "9A", "F minor": "4A", "F# minor": "11A",
    "G minor": "6A", "G# minor": "1A",
}

__all__ = ["detect_key", "camelot_of"]

_PROC = None


def camelot_of(label: str) -> str:
    """Camelot code of a madmom key label such as ``"F# minor"`` (``""`` if unknown)."""
    return _CAMELOT.get(label, "")


def detect_key(audio: np.ndarray, sr: int) -> Tuple[str, float, str]:
    """Return ``(camelot, confidence 0..1, key name)`` for ``audio [n, ch]``.

    Long files are processed in 4-minute chunks whose class probabilities are averaged
    (weighted by chunk length).
    """
    global _PROC
    from madmom.audio.signal import Signal
    from madmom.features.key import CNNKeyRecognitionProcessor, KEY_LABELS
    if _PROC is None:
        _PROC = CNNKeyRecognitionProcessor()
    mono = np.asarray(audio, np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    if len(mono) < sr:
        return "", 0.0, ""
    clen = int(CHUNK_S * sr)
    probs = np.zeros(24)
    for s in range(0, len(mono), clen):
        part = mono[s:s + clen]
        if len(part) < 5 * sr and s > 0:
            break
        p = np.asarray(_PROC(Signal(part, sample_rate=sr, num_channels=1))).reshape(-1)
        probs += p * len(part)
    probs /= probs.sum()
    i = int(np.argmax(probs))
    name = KEY_LABELS[i]
    return camelot_of(name), float(probs[i]), name
