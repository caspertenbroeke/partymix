"""v2 mix engine: beat-accurate placement from bars.json, plan from plan.json"""
import json, subprocess, os, sys, numpy as np, soundfile as sf, pyloudnorm as pyln
from scipy.signal import butter, sosfilt, sosfiltfilt

SR = 44100
B = json.load(open("bars.json"))
P = json.load(open(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "plan.json"))
SET, TR = P["set"], P["tr"]          # set: [key, s, e]; tr: [overlap, tempo, preset, opts]
n = len(SET)
os.makedirs("st2", exist_ok=True)

def src_time(key, bar):
    """source seconds of (fractional) bar index, via the beat list"""
    b = np.array(B[key]["beats"]); f = B[key]["first"]
    x = f + bar * 4
    i = np.clip(np.floor(x).astype(int), 0, len(b) - 2)
    return b[i] + (x - i) * (b[i + 1] - b[i])

# ---- output layout ----
O = [0]
for i in range(n - 1):
    O.append(O[i] + (SET[i][2] - SET[i][1]) - TR[i][0])
TOTAL_BARS = O[-1] + SET[-1][2] - SET[-1][1]
anch = [(0, P["start_tempo"])]
for i in range(n - 1):
    anch += [(O[i + 1], TR[i][1]), (O[i + 1] + TR[i][0], TR[i][1])]
anch.append((TOTAL_BARS + 8, P["end_tempo"]))
ax = np.array([a for a, _ in anch], float); ay = np.array([t for _, t in anch], float)
beat_tempo = np.interp((np.arange((TOTAL_BARS + 8) * 4) + 0.5) / 4, ax, ay)
beat_dur = 60 / beat_tempo
beat_t = np.concatenate([[0], np.cumsum(beat_dur)])
def T(xbar):
    xb = np.asarray(xbar) * 4
    j = np.clip(np.floor(xb).astype(int), 0, len(beat_dur) - 1)
    return beat_t[j] + (xb - j) * beat_dur[j]
TOTAL_SEC = float(T(TOTAL_BARS))

TAIL = 4
def stretch(i):
    pre = None
    for part in ("inst", "voc"):
        pre = stretch_part(i, part)
    return i, pre

def stretch_part(i, part):
    key, s, e = SET[i]
    out = f"st2/{i:02d}_{key}_{part}.wav"
    a = max(float(src_time(key, s)) - 0.05, 0); z = min(float(src_time(key, e + TAIL)), B[key]["beats"][-1] + 0.4)
    raw = f"st2/{i:02d}_{key}_{part}_in.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.5f}", "-to", f"{z:.5f}", "-i", f"inst/{key}_{part}.wav", "-c:a", "pcm_f32le", raw], check=True)
    # also stretch the vocal stem the same way (for checking)
    lines = ["0 0"] if src_time(key, s) - a > 0.001 else []
    pre_out = None
    for q in range(int((e + TAIL - s) * 4) + 1):
        st = float(src_time(key, s + q / 4)) - a
        if st > z - a - 0.01: break
        ot = float(T(O[i] + q / 4) - T(O[i]))
        if q == 0:
            pre = st  # source seconds before bar s; map them before out 0 with same local ratio
            pre_out = pre
        lines.append(f"{int(round(st * SR))} {int(round((ot + pre_out) * SR))}")
    lines = list(dict.fromkeys(lines))
    open(raw + ".map", "w").write("\n".join(lines) + "\n")
    li, lo = map(int, lines[-1].split())
    inlen = sf.info(raw).frames
    outlen = lo + int((inlen - li) * (lo / max(li, 1)))
    subprocess.run(["rubberband", "-q", "-3", "-M", raw + ".map", "-D", f"{outlen / SR:.6f}", raw, out], check=True)
    return pre_out

if __name__ == "__main__" and "stretch" in sys.argv:
    from concurrent.futures import ProcessPoolExecutor
    pres = {}
    with ProcessPoolExecutor(2) as ex:
        for i, pre in ex.map(stretch, range(n)):
            pres[i] = pre; print("stretched", i, SET[i][0], flush=True)
    json.dump(pres, open("st2/pre.json", "w"))
    print("total %d bars, %.1f s = %d:%02d" % (TOTAL_BARS, TOTAL_SEC, TOTAL_SEC // 60, TOTAL_SEC % 60))
