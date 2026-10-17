# PartyMix

**Van een stapel losse nummers naar één doorlopende feestmix, zonder dat je zelf hoeft te mixen.**

Je geeft PartyMix 10 tot 30 nummers. PartyMix zoekt een goede volgorde en legt de beats van elk nummer
precies op elkaar. Daarna maakt het vloeiende overgangen: de bas van het volgende nummer komt binnen
op het moment dat de vorige wegvalt, en twee zangstemmen klinken nooit door elkaar. Het resultaat is
één audiobestand (bijvoorbeeld een mix van 30 minuten) dat je zo op een feest kunt afspelen.

Alles gebeurt op je eigen computer. Er wordt niets geüpload.

---

## Downloaden en installeren

1. Ga naar de pagina **[Releases](https://github.com/caspertenbroeke/partymix/releases)**.
2. Download bij de nieuwste versie **`PartyMix-Setup-x.y.z.exe`**.
3. Start het bestand en volg de stappen. Je hebt **geen beheerdersrechten** nodig: PartyMix wordt
   alleen voor jouw Windows-account geïnstalleerd (in `%LOCALAPPDATA%\Programs\PartyMix`).
   Je krijgt een snelkoppeling in het Startmenu en, als je dat wilt, op het bureaublad.

> **Windows zegt "Windows heeft uw pc beschermd"?**
> PartyMix is (nog) niet digitaal ondertekend, en daarom geeft Windows SmartScreen deze waarschuwing.
> Klik op **Meer info** en daarna op **Toch uitvoeren**.

**Liever niets installeren?** Download dan `PartyMix-x.y.z-portable-win64.zip`, pak het uit
(bijvoorbeeld op een USB-stick) en start `PartyMix.exe` in die map.

**Verwijderen** gaat via *Instellingen → Apps → PartyMix → Verwijderen*. De analyse-cache wordt
daarbij ook opgeruimd. Je gemaakte mixen blijven staan.

---

## Zo gebruik je het

1. **Sleep je nummers erin.** Sleep 10–30 audiobestanden (of een hele map) naar het PartyMix-venster.
   PartyMix leest MP3, WAV, FLAC, M4A/AAC, OGG/Opus en AIFF.
2. **Klik op _Analyseren_.** Per nummer bepaalt PartyMix het tempo (BPM), de beats en maten, de
   toonsoort, waar de zang zit en hoe energiek elk stuk is. Je ziet per nummer de voortgang.
   Een nummer dat al eens is geanalyseerd, hoeft daarna niet opnieuw (dat staat in de cache).
3. **Kijk naar de _Volgorde_.** PartyMix stelt een volgorde voor: rustig beginnen, langzaam opbouwen,
   een piek rond driekwart en dan een korte finale. Het let ook op tempo en toonsoort. Wil je een
   bepaald nummer als eerste of als laatste? Zet het vast, of sleep nummers zelf naar een andere plek.
4. **Klik op _Mix maken_.** Kies de doelduur (standaard 30 minuten), het formaat en de map.
   PartyMix rekt elk nummer heel licht uit tot hetzelfde tempo, maakt de overgangen en brengt de mix
   op een goed volume. Daarna kun je de map openen of de mix meteen afspelen.

### Uitvoerformaten

| Formaat | Kwaliteit | Wanneer kiezen |
|---|---|---|
| **FLAC 24-bit** | lossless, ± half zo groot als WAV | standaard; prima voor elke afspeelsoftware die FLAC kent |
| **WAV 24-bit** | lossless, groot (± 15 MB per minuut) | als je software geen FLAC kent |
| **MP3 320 kbps** | lossy, klein | voor je telefoon, WhatsApp of een speaker met USB-stick |

---

## Tips voor het beste resultaat

* **Bronbestanden bepalen de kwaliteit.** Met **WAV- of FLAC-bronnen** is de hele keten echt lossless.
  Met **MP3-bronnen** blijft de mix op MP3-kwaliteit, ook als je naar FLAC of WAV exporteert.
  Een groter bestand maakt het geluid niet beter. Gebruik zo mogelijk 320 kbps-MP3's of beter.
* **Kies nummers met een vergelijkbaar tempo.** Binnen ongeveer ±8 % (bijvoorbeeld 120–130 BPM) klinken
  de overgangen het natuurlijkst. Grotere sprongen kunnen, maar dan wordt een nummer merkbaar sneller
  of trager.
* **Dansbare nummers met een vaste beat** werken het best: house, dance, pop, Nederlandstalige
  feesthits, latin. Nummers zonder vaste maat (live-opnames, ballads met tempowisselingen) gaan ook,
  maar daar zijn de overgangen lastiger.
* **Gebruik de originele versies.** Kies geen radio-edits die abrupt beginnen of eindigen, en geen
  YouTube-rips met praat of geluidseffecten erin.
* **Geef het de tijd.** De eerste analyse is het zwaarste werk. Een tweede mix met dezelfde nummers
  is veel sneller klaar.
* **Laat je laptop aan de stekker** tijdens het analyseren en mixen. Op batterij gaat het vaak een
  stuk langzamer.

---

## Systeemeisen

| | Minimaal | Aanbevolen |
|---|---|---|
| Besturingssysteem | Windows 10 of 11, **64-bit** | Windows 11 |
| Processor | 4 cores (bijv. Intel Core i5 8e generatie / Ryzen 5 2000) | 6–8 cores of meer |
| Werkgeheugen | 8 GB | 16 GB (voor mixen langer dan 30 minuten) |
| Schijfruimte | ± 1 GB voor het programma | + ± 50–100 MB cache per nummer, plus ruimte voor je mixen |
| Videokaart | niet nodig | – |

PartyMix werkt volledig offline. Een videokaart wordt niet gebruikt: alles draait op de processor.

### Hoe lang duurt het?

Dit zijn **grove schattingen** voor een gemiddelde laptop met 4–8 cores. De eerste keer is het meeste
werk het scheiden van zang en muziek.

| Stap | Per nummer van ± 4 min | 20 nummers |
|---|---|---|
| Analyseren (eerste keer) | 1–4 minuten | 20–60 minuten |
| Analyseren (al in cache) | een paar seconden | < 1 minuut |
| Mix maken (30 min) | – | 5–15 minuten |

Een snelle desktop-pc is vaak 2× sneller, een oudere laptop kan 2× langzamer zijn.

---

## Veelgestelde vragen

**Het analyseren duurt heel lang. Is het vastgelopen?**
Waarschijnlijk niet. Het scheiden van zang en muziek kost veel rekenkracht. Zolang de voortgangsbalk
af en toe verder gaat, is PartyMix bezig. Je kunt ondertussen gewoon iets anders doen.

**Mijn computer wordt warm en de ventilator draait hard.**
Dat is normaal: PartyMix gebruikt alle cores van je processor.

**Kan ik de volgorde zelf bepalen?**
Ja. Sleep de nummers in de lijst naar de volgorde die jij wilt, of zet alleen het eerste en/of laatste
nummer vast. PartyMix regelt dan de rest.

**Eén nummer klinkt raar of te snel in de mix.**
Dat nummer heeft waarschijnlijk een heel ander tempo dan de nummers ernaast, of geen vaste beat.
Haal het eruit of zet het op een andere plek in de lijst.

**Is mijn mix lossless?**
Alleen als je bronnen lossless zijn (WAV/FLAC) en je exporteert naar WAV of FLAC. Intern rekent
PartyMix met hoge precisie (32/64-bit float) en voegt het geen extra kwaliteitsverlies toe. Een MP3 als
bron blijft wel een MP3: verloren details komen niet terug.

**Waar laat PartyMix zijn tijdelijke bestanden?**
In `%LOCALAPPDATA%\PartyMix\cache`. Die map mag je gerust leegmaken: de nummers worden dan de volgende
keer opnieuw geanalyseerd. Bij het verwijderen van PartyMix wordt de map automatisch opgeruimd.

**Mag ik PartyMix gebruiken voor een betaald optreden?**
Nee. De modellen voor beat- en toonsoortherkenning (van madmom) mogen alleen **niet-commercieel**
worden gebruikt. Zie de licenties hieronder.

**Werkt het op een Mac of Linux?**
Er is alleen een Windows-versie. Ontwikkelaars kunnen het wel vanuit de broncode draaien (zie hieronder).

**Ik heb een probleem gevonden.**
Meld het via [Issues](https://github.com/caspertenbroeke/partymix/issues). Zet erbij welke versie je
gebruikt en wat je deed.

---

## Licentie

PartyMix is gratis en open source onder de **GNU GPL v3.0 of later** (zie [`LICENSE`](LICENSE)).

PartyMix gebruikt onder andere:
**madmom** (code BSD, **modellen CC BY-NC-SA 4.0 → alleen niet-commercieel gebruik**),
**Rubber Band** (GPL), **FFmpeg** (GPL-build), het **UVR-MDX-NET** zangscheidingsmodel van
Ultimate Vocal Remover (MIT), **ONNX Runtime** (MIT) en **Qt/PySide6** (LGPL).
Alle details en bronvermeldingen staan in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Voor de rechten op de muziek die je mixt en (in het openbaar) afspeelt, ben je zelf verantwoordelijk.

---

## Voor ontwikkelaars

<details>
<summary>Bouwen, testen en releasen</summary>

### Lokaal draaien (Python 3.11)

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                   # numpy, scipy, PySide6, onnxruntime, ... (geen torch)
pip install --no-build-isolation -r requirements-madmom.txt   # madmom uit git; vereist C-compiler (MSVC op Windows)
python -m pytest                                      # tests die lokale audio nodig hebben slaan zichzelf over
python -m partymix                                    # GUI
python -m partymix all nummers/ -o mix.flac --minutes 30   # CLI
```

Buiten de bundel worden `ffmpeg`, `ffprobe` en `rubberband` op de `PATH` gezocht (zie `partymix/paths.py`).

### Windows-build

Alles staat in `.github/workflows/build-windows.yml` en draait bij elke push, bij een pull request of
met de hand (*workflow_dispatch*):

1. dependencies installeren en madmom als wheel bouwen (gecachet per commit);
2. `pytest` (headless Qt via `QT_QPA_PLATFORM=offscreen`);
3. `python packaging/fetch_tools.py`: downloadt ffmpeg/ffprobe (gyan.dev 9.0.2 essentials, GPLv3),
   Rubber Band 4.0.0 (+ `sndfile.dll`) en `UVR-MDX-NET-Inst_HQ_3.onnx`. Elk bestand wordt gecontroleerd
   met een **vaste SHA-256**. Het resultaat komt in `packaging/bin`, `packaging/models` en
   `packaging/licenses`;
4. PyInstaller (onedir) met `packaging/partymix.spec`. Dat levert `dist/PartyMix/PartyMix.exe` (venster)
   en `partymix-cli.exe` (console) op;
5. **zelftest** van de bevroren app (zie hieronder), daarna de portable zip, de Inno Setup-installer
   (`packaging/installer.iss` → `PartyMix-Setup-<versie>.exe`) en een test van de installer:
   stil installeren, zelftest, verwijderen;
6. artifacts uploaden. Bij een tag `v*` volgt ook een **GitHub Release** met de installer en de zip.

**Een release maken:** zet `__version__` in `partymix/__init__.py` (bijv. `0.2.0`), commit, en push een
tag `v0.2.0`. Komen tag en versie niet overeen, dan stopt de build. Een tag met `-` (bijv. `v0.2.0-rc1`)
wordt als pre-release gepubliceerd.

### Zelftest (`--selftest`)

```
PartyMix.exe --selftest [--quick] [--log BESTAND] [--keep MAP]
partymix-cli.exe --selftest ...                  # zelfde, met console-uitvoer
python packaging/partymix_selftest.py ...        # vanuit de broncode
```

De zelftest wordt in de launcher (`packaging/launcher.py`) afgevangen en draait
`packaging/partymix_selftest.py`. Die doet het volgende:

1. alle `partymix`-modules importeren, plus numpy/scipy/madmom/onnxruntime/soundfile/pyloudnorm/PySide6;
2. ffmpeg, ffprobe en rubberband vinden via `partymix.paths` en uitvoeren (`-version`);
3. het ONNX-model laden en één inferentie op stilte draaien (onnxruntime CPU);
4. de madmom-netwerken laden (downbeat + key) en de beats van een click-track van 12 s vinden (120 BPM);
5. een `QApplication` starten (controleert de Qt-platformplugin);
6. (niet met `--quick`) twee synthetische house-tracks maken (128 en 125 BPM) en
   `partymix all a.wav b.wav -o mix.flac --minutes 2` draaien. Daarna wordt gecontroleerd of er een
   geldige mix van minstens 30 s is uitgekomen.

Exitcode 0 betekent dat alle stappen geslaagd zijn. Het log gaat naar stdout, naar `--log` en naar
`$PARTYMIX_SELFTEST_LOG`. De cache wordt naar een tijdelijke map omgeleid.
De zelftest gebruikt alleen de publieke CLI (`partymix.cli.main`).

### Een tool of model bijwerken

Pas de URL aan in `packaging/fetch_tools.py` en draai `python packaging/fetch_tools.py --print-hashes`.
Zet daarna de nieuwe SHA-256 in het bestand en werk `THIRD_PARTY_NOTICES.md` bij.

</details>
