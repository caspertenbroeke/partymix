# Status / handoff (lead) – laatst bijgewerkt 2026-10-17 19:55

Doel: Windows-app die precies doet wat "Half uur mix v2" deed (zie reference/PLAN_NOTES.md), downloadbaar via GitHub Releases.

## Klaar
- `partymix/model.py`, `paths.py`, `ARCHITECTURE.md` (contracten).
- Planner (`partymix/planner/`): order_tracks, pair_score, plan_mix(+report), tempo_in bij echo_cut, MixPlan.notes. 16 tests groen.
  Op de 14 echte tracks: 30:06 bij doel 30 min, keuzes ≈ handplan v2.

## Was nog bezig (agents, check of hun bestanden af zijn — WIP gecommit)
- analysis (`partymix/audio_io.py`, `partymix/analysis/*`): madmom beats/downbeats + grid-fit, key, ONNX vocal separation (port van audio-separator, zonder torch), per-bar features, cache.
- render (`partymix/render/*`): layout (moet tempo_in ondersteunen), rubberband timemap stretch, mixer (port mix2.py), master (limiter, FLAC/WAV 24-bit, MP3), verify.
  Validatie: v2 reproduceren uit /home/claude/mix/inst + reference/plan.json en vergelijken met /home/claude/mix/mix2_raw.wav.
- GUI/CLI (`partymix/gui/*`, `cli.py`, `__main__.py`, `worker.py`): PySide6, Nederlands, multiprocessing worker.
- packaging (`packaging/*`, `.github/workflows/*`, README, LICENSE GPL-3.0, THIRD_PARTY_NOTICES): PyInstaller + Inno Setup op windows-latest, release op tag v*.
  `PartyMix.exe --selftest` moet door GUI/CLI geïmplementeerd worden.

## Volgende stappen
1. Per module: pytest draaien, afmaken wat half is (zie TODO's/tests).
2. Integratie: `python -m partymix all <14 tracks> -o test_out/mix.flac --minutes 30`, daarna verify (beats on grid ≥95 % per track, zoals v2).
3. Push → GitHub Actions build fixen tot groen → tag v0.1.0 → release met installer.
4. Gebruiker (Casper) melden in het Nederlands met downloadlink.

## Lokale testdata (NIET in git, auteursrecht)
Container: /home/claude/mix/wav (14 tracks), /home/claude/mix/inst (*_inst.wav/*_voc.wav stems), /home/claude/mix/mm (madmom),
/home/claude/mix/mix2_raw.wav (v2 pre-master). Als de container weg is: bronbestanden opnieuw ophalen uit
de map "C:\Users\caspe\Music\Half uur mix" op zijn computer (device_stage_files), de rest is reproduceerbaar.
