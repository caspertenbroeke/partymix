import sys, json, numpy as np, soundfile as sf, pyloudnorm as pyln, subprocess
from scipy.signal import butter, sosfilt, sosfiltfilt
import engine as E

SR = E.SR; n = E.n
PRE = {int(k): v for k, v in json.load(open("st2/pre.json")).items()}
def sm(u):  # smoothstep
    u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)
def rp(u, a, b): return np.clip((u - a) / (b - a), 0, 1)

def preset(name, role, u, opt):
    """vol, bass, filt, fx, vocal gain"""
    one, zero = np.ones_like(u), np.zeros_like(u)
    sw = opt.get("swap", 0.5)
    if role == "A":
        vol = one.copy(); bass = (u < sw).astype(float) if sw > 0 else zero.copy()
        filt = zero.copy(); fx = zero.copy()
        voc = (u < sw).astype(float) if opt.get("voc", 1) else one.copy()
        if name == "swap_hp": filt = 0.5 * sm(rp(u, 0.5, 1))
        if name == "editor": filt = -0.6 * sm(rp(u, 0.5, 1)); fx = 0.45 * rp(u, 0.75, 1)
        if name == "xfade": vol = np.cos(sm(rp(u, sw, 1)) * np.pi / 2)
        if name == "echo_out": fx = 0.6 * rp(u, 0.5, 1)
        return vol, bass, filt, fx, voc
    fin = opt.get("fadein", 0.25)
    vol = sm(rp(u, 0, fin)) if fin > 0 else one.copy()
    bass = (u >= sw).astype(float)
    voc = (u >= sw).astype(float) if opt.get("voc", 1) else one.copy()
    lpd = opt.get("lpdur", max(sw, 0.25) if sw < 1 else 0.5)
    filt = -0.65 * (1 - sm(rp(u, 0, lpd))) if opt.get("lpin") else zero.copy()
    return vol, bass, filt, zero.copy(), voc

def lufs(x):
    m = pyln.Meter(SR); blk = SR * 3
    segs = [x[i:i + blk] for i in range(0, len(x) - blk, blk)]
    rms = np.array([np.sqrt((s ** 2).mean()) for s in segs])
    keep = np.concatenate([segs[i] for i in np.argsort(rms)[len(rms) // 3:]])
    return m.integrated_loudness(keep)

def tv_filter(x, fb, blk):
    y = np.empty_like(x); zi = None; mode = 0
    for b in range(len(fb)):
        seg = x[b * blk:(b + 1) * blk]
        if len(seg) == 0: break
        f = fb[b]
        if abs(f) < 0.01:
            y[b * blk:b * blk + len(seg)] = seg; zi = None; mode = 0; continue
        if f < 0: fc = 18000 * (300 / 18000) ** (-f); sos = butter(2, fc, "low", fs=SR, output="sos"); m = -1
        else: fc = 30 * (900 / 30) ** f; sos = butter(2, fc, "high", fs=SR, output="sos"); m = 1
        if zi is None or m != mode: zi = np.zeros((1, 2, 2))
        out, zi = sosfilt(sos, seg, axis=0, zi=zi); y[b * blk:b * blk + len(seg)] = out; mode = m
    return y

def echo(x, send, D, fb=0.5):
    inp = x * send[:, None]; y = np.zeros_like(x)
    for s in range(0, len(x), D):
        e = min(s + D, len(x)); y[s:e] = inp[s:e] + (fb * y[s - D:e - D] if s >= D else 0)
    return y

total_len = int((E.TOTAL_SEC + 8) * SR)
master = np.zeros((total_len, 2), np.float32)
LP = butter(4, 160, "low", fs=SR, output="sos")
BLK = 512; cues = []
for i, (key, s, e) in enumerate(E.SET):
    xi = sf.read(f"st2/{i:02d}_{key}_inst.wav", dtype="float32", always_2d=True)[0].astype(np.float64)
    xv = sf.read(f"st2/{i:02d}_{key}_voc.wav", dtype="float32", always_2d=True)[0].astype(np.float64)
    m_ = min(len(xi), len(xv)); xi, xv = xi[:m_], xv[:m_]
    x = xi + xv
    pre = PRE[i]; L = e - s
    t0 = float(E.T(E.O[i]))
    loc = lambda bar: float(E.T(E.O[i] + bar) - t0) + pre        # local file seconds of local bar
    end_s = loc(L)
    x = x[: int((loc(L + 3)) * SR)]; xv = xv[: len(x)]; xi = xi[: len(x)]
    seg_body = x[max(0, int(loc(0) * SR)): int(end_s * SR)]
    gain = 10 ** ((-14 - lufs(seg_body)) / 20); xi *= gain; xv *= gain
    N = len(x); nb = N // BLK + 1; tb = (np.arange(nb) + 0.5) * BLK / SR
    vol = np.ones(nb); bass = np.ones(nb); filt = np.zeros(nb); fx = np.zeros(nb); vg = np.ones(nb)
    vol[(tb < loc(0)) | (tb >= end_s)] = 0
    def apply(ta, tz, name, role, opt):
        m = (tb >= ta) & (tb < tz); u = (tb[m] - ta) / (tz - ta)
        v, b_, f_, s_, g_ = preset(name, role, u, opt)
        vol[m] *= v; bass[m] *= b_; filt[m] += f_; fx[m] = np.maximum(fx[m], s_); vg[m] *= g_
    if i > 0:
        ov, _, name, opt = E.TR[i - 1]
        if ov > 0: apply(loc(0), loc(ov), name, "B", opt)
    if i < n - 1:
        ov, _, name, opt = E.TR[i]
        if ov > 0: apply(loc(L - ov), end_s, name, "A", opt)
        elif name == "echo_cut":
            a2 = loc(L - 1); m = (tb >= a2) & (tb < end_s); u = (tb[m] - a2) / (end_s - a2)
            fx[m] = 1.0; vol[m] *= 1 - sm(u); bass[m] *= 1 - sm(rp(u, 0, 0.5))
    else:
        fb_ = P_END = E.P.get("end_fade", 8)
        a2 = loc(L - fb_); m = (tb >= a2) & (tb < end_s); vol[m] *= 1 - sm((tb[m] - a2) / (end_s - a2))
    def up(env, ms=10):
        k = max(1, int(ms / 1000 * SR / BLK))
        env = np.convolve(np.pad(env, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
        return np.interp(np.arange(N), (np.arange(len(env)) + 0.5) * BLK, env)
    x = xi + xv * up(vg, 60)[:, None]
    low = sosfiltfilt(LP, x, axis=0)
    x = (x - low) + low * up(bass, 12)[:, None]
    if np.abs(filt).max() > 0.01: x = tv_filter(x, filt, BLK)
    out = x * up(vol, 6)[:, None]
    if fx.max() > 0:
        tempo_here = E.TR[i][1] if i < n - 1 else E.P["end_tempo"]
        send = up(fx, 20) * up(((tb >= loc(0)) & (tb < end_s)).astype(float), 5)
        wet = echo(x * up(vol, 6)[:, None] + 0, send, int(0.75 * 60 / tempo_here * SR))
        tail = np.clip(1 - (np.arange(N) / SR - end_s) / (loc(L + 2) - end_s), 0, 1)
        out += 0.55 * wet * tail[:, None]
    st = int((t0 - pre) * SR)
    a_ = max(0, -st); seg = out[a_: total_len - st].astype(np.float32)
    master[st + a_: st + a_ + len(seg)] += seg
    cues.append((t0, key)); print("mixed", i, key, "%d:%02d" % (t0 // 60, t0 % 60), flush=True)

master = master[: int((E.TOTAL_SEC + 3) * SR)]
sf.write("mix2_raw.wav", master, SR, subtype="FLOAT")
json.dump(cues, open("cues2.json", "w"))
l = pyln.Meter(SR).integrated_loudness(master)
name = sys.argv[2] if len(sys.argv) > 2 else "Half uur mix v2.mp3"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "mix2_raw.wav", "-af",
                f"volume={-9.5 - l:.2f}dB,alimiter=limit=0.89:attack=3:release=80:level=disabled",
                "-c:a", "libmp3lame", "-b:a", "320k", name], check=True)
print("done", name)
