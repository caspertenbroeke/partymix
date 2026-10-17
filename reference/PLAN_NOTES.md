# How the v2 plan (`plan.json`) was chosen by hand — the planner must automate this

Data: `maps_v2.txt` (per bar: B = bass level 0–9, V = vocal level 0–9, groups of 8 bars, bar 0 = first downbeat),
`grid2.json` (bpm/first downbeat), `bars.json` (beats, per-bar features), `vox.json`.
v1 failed because 4 tracks were half a beat off (bad downbeat detection). v2 fixed that with madmom and
the user called v2 "insane goed". v2 verification: 95–100 % of detected beats/downbeats on the intended grid.

## Rules I applied
1. **Phrases**: sections change on multiples of 8 bars from a per-track phrase offset (often 0, sometimes 4).
   All in/out points and swap points are on phrase boundaries (or at least on 4-bar boundaries).
2. **Incoming window** `[s, s+ov)`: ends exactly where the incoming track's first big section starts
   (bass enters / drop / chorus). Typical: intro bars with little bass. e.g. Die Young in at bar 4, its drop at 20 → ov 16.
   Levels in at bar 4, bass at 12 → ov 8 with swap at 1.0 (bass handed over at the very end).
3. **Outgoing window** `[e-ov, e)`: the end of a section of the outgoing track (end of a drop before a break, or a break/outro).
   Never cut in the middle of a phrase. Examples: World Hold On out over its final drop 76–92;
   Levels out during its long break 52–60 (no bass) while Rampeneren's instrumental intro comes in.
4. **Swap point** (bass + vocal hand-over) = where the incoming bass starts relative to the window
   (0.5 if the intro already has bass, 1.0 if bass starts right at the window end, 0.0 if the outgoing has no bass).
5. **Vocal hand-over** (most important for sound): incoming vocals muted before the swap, outgoing vocals muted after it.
   This is done with the separated stems, so vocal intros/outros are fine.
6. **Transition length**: 16 bars when both have long intro/outro material, 8 bars otherwise; 0 + `echo_cut` when tempo/key/energy jump
   (Brokko Maken 135 → Axel F 138 intro).
7. **Tempo**: transition tempo between the two native tempos (closer to the outgoing). Tracks never stretched more than ~2.7 %.
   Solo parts ramp linearly from previous transition tempo to the next.
8. **Length budget**: target 30 min ≈ 980 bars at ~131 BPM. With 14 tracks each track plays ~70–95 bars incl. overlaps.
   First pass came out at 16 min; lengthened by using later out-points (keep whole drops/choruses).
9. **Presets** used: `xfade` (equal power after swap, B fades in over first half, LPF on B), `swap_hp` (gentle HPF on A in 2nd half),
   `swap`, `editor` (B fade-in, A LPF + echo in last quarter = Spotify "fade in / center bass swap / LPF out / reverb cut end"),
   `echo_out` (A echo in 2nd half), `echo_cut` (no overlap; A echoes out over its last bar, B starts hard on the 1).

## The v2 plan in words (source bars)
| # | track | bpm | play | out window → next in window | preset |
|---|---|---|---|---|---|
| 1 | World Hold On | 127 | 4–92 | 76–92 → SL 16–32 | xfade, fadein .5, lpin |
| 2 | Stereo Love | 127 | 16–96 | 80–96 → DY 4–20 | swap_hp, lpin |
| 3 | Die Young | 128 | 4–84 | 76–84 → LV 4–12 | swap @1.0, lpin |
| 4 | Levels | 126 | 4–60 | 52–60 (break) → RA 0–8 | swap, fadein .5, lpin |
| 5 | Rampeneren | 125 | 0–88 | 80–88 → ITT 4–12 | editor @1.0 |
| 6 | It's That Time | 130 | 4–92 | 76–92 → MEM 16–32 | swap, fadein .25 |
| 7 | Memories (ext) | 130 | 16–96 | 88–96 → KB 0–8 | editor |
| 8 | Kaolo Beat | 132 | 0–56 | 48–56 → YQF 0–8 | swap_hp, lpin |
| 9 | Y Que Fue | 133 | 0–72 | 64–72 → LM 0–8 | swap @0 (A bass off), lpin |
| 10 | La Mamá | 129.7 | 0–64 | 56–64 → SCH 0–8 | echo_out @1.0 |
| 11 | Schudden | 135.1 | 0–96 | 88–96 → BM 0–8 | swap |
| 12 | Brokko Maken | 135 | 0–64 | echo_cut | — |
| 13 | Axel F | 138 | 0–88 | 80–88 → VB 8–16 | swap @1.0 |
| 14 | We Like To Party | 136 | 8–124 | end fade 8 bars | — |
Tempos at transitions: 127, 127.5, 127, 125.5, 127, 130, 131, 132.5, 131.5, 132.5, 135, 136(cut), 137; end 136.5.
