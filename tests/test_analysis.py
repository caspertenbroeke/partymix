"""Unit tests for the analysis package (synthetic audio, no real music needed)."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from partymix import audio_io, paths
from partymix.analysis import beats as B
from partymix.analysis import features as F
from partymix.analysis import key as K
from partymix.analysis import vocals as V

SR = 44100


def synth_track(bpm: float = 128.0, offset: float = 0.37, bars: int = 40, sr: int = SR,
                pickup_beats: int = 0) -> np.ndarray:
    """Four-on-the-floor test track: kick every beat, snare on 2 & 4, hats on 8ths, a bass line
    and chord that change on every downbeat (beat 1 is also accented), crash every 4 bars.
    The first downbeat is at ``offset + pickup_beats * period`` seconds."""
    rng = np.random.default_rng(1)
    per = 60.0 / bpm
    n_beats = bars * 4 + pickup_beats
    n = int((offset + (n_beats + 2) * per) * sr)
    y = np.zeros(n, np.float64)

    def add(sig: np.ndarray, t: float) -> None:
        i = int(round(t * sr))
        if i >= n:
            return
        m = min(len(sig), n - i)
        y[i:i + m] += sig[:m]

    tk = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 120 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 9)
    tn = np.arange(int(0.15 * sr)) / sr
    snare = rng.standard_normal(len(tn)) * np.exp(-tn * 25) * 0.5
    th = np.arange(int(0.04 * sr)) / sr
    hat = rng.standard_normal(len(th)) * np.exp(-th * 120) * 0.15
    tc = np.arange(int(1.5 * sr)) / sr
    crash = rng.standard_normal(len(tc)) * np.exp(-tc * 3) * 0.25
    roots = [55.0, 43.65, 49.0, 41.2]          # A, F, G, E – one per bar
    for j in range(n_beats):
        t = offset + j * per
        bar, beat = divmod(j - pickup_beats, 4)
        add(kick * (1.0 if beat == 0 else 0.7), t)
        add(hat, t + per / 2)
        if beat in (1, 3):
            add(snare, t)
        if j >= pickup_beats and beat == 0:
            f0 = roots[bar % 4]
            tb = np.arange(int(4 * per * sr)) / sr
            env = np.minimum(1, tb * 50) * np.exp(-tb * 0.3)
            chord = sum(np.sin(2 * np.pi * f0 * m * tb) for m in (2, 2.5198, 2.9966, 4))
            add((0.35 * np.sin(2 * np.pi * f0 * tb) + 0.08 * chord) * env, t)
            if bar % 4 == 0:
                add(crash, t)
    y /= np.abs(y).max() * 1.12
    return np.stack([y, y], axis=1).astype(np.float32)


def _phase_err(t: float, ref: float, bar_len: float) -> float:
    d = (t - ref) % bar_len
    return min(d, bar_len - d)


# ----------------------------------------------------------------------------- beats
@pytest.mark.parametrize("bpm,offset,pickup", [(128.0, 0.37, 0), (124.0, 0.21, 2)])
def test_synthetic_bpm_and_downbeat(bpm, offset, pickup):
    x = synth_track(bpm, offset, bars=40, pickup_beats=pickup)
    r = B.track_beats(x, SR)
    per = 60.0 / bpm
    assert r.constant_tempo
    assert abs(r.bpm - bpm) < 0.05
    first = offset + pickup * per
    assert _phase_err(r.first_downbeat, first, 4 * per) < 0.03
    assert r.beats[0] > -0.05 and r.beats[0] < per          # grid extended back to ~0 s
    assert r.beats[-1] > len(x) / SR - per - 0.02           # … and to the end of the file
    assert np.allclose(np.diff(r.beats), per, atol=1e-6)


def _raw(bpm, n, phase=0, jitter=0.005, start=0.5, seed=0):
    rng = np.random.default_rng(seed)
    per = 60 / bpm
    t = start + np.arange(n) * per + rng.normal(0, jitter, n)
    pos = (np.arange(n) + phase) % 4 + 1
    return np.stack([t, pos], 1)


def test_fit_grid_constant_with_gaps_and_outliers():
    raw = _raw(130.0, 400, phase=2)
    raw = np.delete(raw, [50, 51, 200], axis=0)          # missing beats
    raw[300, 0] += 0.2                                   # outlier
    r = B.fit_grid(raw, 200.0)
    assert r.constant_tempo and abs(r.bpm - 130.0) < 0.02
    per = 60 / 130
    # beat index 2 had position 1  ->  downbeat at start + 2 per (mod bar)
    assert _phase_err(r.first_downbeat, 0.5 + 2 * per, 4 * per) < 0.01
    assert 0 <= r.first_downbeat_index <= 3


def test_fit_grid_non_constant_keeps_madmom_beats():
    a = _raw(128.0, 150, jitter=0.002)
    b = _raw(132.0, 150, jitter=0.002, start=a[-1, 0] + 60 / 132, seed=1)
    b[:, 1] = (np.arange(150) + 150) % 4 + 1
    r = B.fit_grid(np.vstack([a, b]), b[-1, 0] + 1.0)
    assert not r.constant_tempo
    assert abs(r.first_downbeat - a[0, 0]) < 0.01
    assert len(r.beats) >= 300


def test_stitch_relabels_and_dedups():
    full = _raw(125.0, 200, jitter=0.0)
    a = full[full[:, 0] < 60]
    b = full[full[:, 0] >= 50].copy()
    b[:, 1] = (b[:, 1] % 4) + 1                          # other chunk labelled one beat off
    s = B._stitch(a, b, 50.0, 10.0)
    assert len(s) == len(full)
    assert np.array_equal(s[:, 1], full[:, 1])


# ----------------------------------------------------------------------------- audio io
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_decode_unicode_and_tags(tmp_path):
    x = synth_track(bars=4)
    wav = tmp_path / "src.wav"
    import soundfile as sf
    sf.write(wav, x[:, 0], SR)                           # mono source
    out = tmp_path / "Ke$ha - La Mamá de la Mamá (Official Video).mp3"
    subprocess.run([paths.ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-ar", "48000", str(out)], check=True)
    y, sr = audio_io.decode(out)
    assert sr == 44100 and y.ndim == 2 and y.shape[1] == 2 and y.dtype == np.float32
    assert abs(len(y) / sr - len(x) / SR) < 0.1
    assert audio_io.read_title_artist(out) == ("La Mamá de la Mamá", "Ke$ha")
    tagged = tmp_path / "x.mp3"
    subprocess.run([paths.ffmpeg(), "-v", "error", "-y", "-i", str(wav), "-metadata", "title=Schudden",
                    "-metadata", "artist=Def Rhymz", str(tagged)], check=True)
    assert audio_io.read_title_artist(tagged) == ("Schudden", "Def Rhymz")


def test_flac_roundtrip_with_headroom(tmp_path):
    x = (np.random.default_rng(0).standard_normal((SR, 2)) * 0.3).astype(np.float32)
    x[100] = 1.45                                        # mp3-style over
    g = audio_io.write_flac(tmp_path / "a.flac", x)
    assert g == 0.5
    y, sr = audio_io.read_audio(tmp_path / "a.flac")
    assert sr == SR and np.abs(y - x).max() < 2e-6


# ----------------------------------------------------------------------------- vocals (STFT port)
def test_stft_roundtrip():
    x = np.random.default_rng(0).standard_normal((2, V.HOP * (V.DIM_T - 1))).astype(np.float32)
    S = V._stft(x)
    assert S.shape == (2, V.N_FFT // 2 + 1, V.DIM_T)
    assert np.abs(V._istft(S, x.shape[1]) - x).max() < 1e-4


@pytest.mark.skipif(not paths.model_path().exists(), reason="ONNX model not available")
def test_separation_sums_to_mix():
    x = synth_track(bars=4)[: 8 * SR]
    inst, voc = V.separate(x)
    assert inst.shape == x.shape == voc.shape
    assert np.abs(inst + voc - x).max() < 1e-5
    # a drum/bass/chord track has (almost) no vocals
    assert 10 * np.log10((voc ** 2).mean() / (x ** 2).mean()) < -10


# ----------------------------------------------------------------------------- key / features
def test_key_detection_runs():
    cam, conf, name = K.detect_key(synth_track(bars=8), SR)
    assert cam in K._CAMELOT.values() and 0 < conf <= 1 and name
    assert K.camelot_of("A minor") == "8A" and K.camelot_of("C major") == "8B"


def test_features_sections_phrase():
    bpm, per = 128.0, 60 / 128
    x = synth_track(bpm, 0.0, bars=40)
    # mute the bass (and quieten) for bars 0-7 and 24-31 -> intro / break structure
    bar = 4 * per
    for a, z in ((0, 8), (24, 32)):
        i, j = int(a * bar * SR), int(z * bar * SR)
        seg = x[i:j, 0].astype(np.float64)
        from scipy.signal import butter, sosfiltfilt
        hp = sosfiltfilt(butter(4, 300, "hp", fs=SR, output="sos"), seg) * 0.5
        x[i:j] = hp[:, None].astype(np.float32)
    beats = list(np.arange(0, len(x) / SR - 0.01, per))
    bounds = F.bar_bounds(beats, 0, len(x) / SR)
    assert len(bounds) == (len(beats)) // 4
    bf = F.bar_features(x, SR, bounds)
    assert len(bf.bass) == len(bounds)
    assert np.mean(bf.bass[8:24]) - np.mean(bf.bass[:8]) > 15
    off = F.phrase_offset(bf.bass, bf.rms, bf.mid)
    assert off == 0
    secs = F.sections(bf.bass, bf.rms, bf.high, off)
    kinds = [s.kind for s in secs]
    assert kinds[0] in ("intro", "build") and "drop" in kinds and "break" in kinds
    assert secs[0].start_bar == 0 and secs[-1].end_bar == len(bounds)
    lufs = F.loudness_lufs(x, SR)
    assert -30 < lufs < 0
    e = F.energy_score(lufs, bf.bass, bpm, F.onset_rate(x, SR))
    assert 1 <= e <= 10


def test_phrase_offset_4():
    rng = np.random.default_rng(0)
    bass = np.full(64, -20.0) + rng.normal(0, 0.5, 64)
    for a in (4, 20, 36, 52):
        bass[a:a + 8] = -40
    voc = np.full(64, -30.0)
    assert F.phrase_offset(bass, voc) == 4
