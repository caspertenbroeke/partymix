"""PartyMix main window."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QApplication, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                               QVBoxLayout, QWidget)

from partymix.cli import expand_inputs, fmt_time, is_audio, tracklist_entries, tracklist_text, AUDIO_EXTS
from partymix.gui import theme
from partymix.gui.backend import Backend
from partymix.gui.models import FAILED, READY, RUNNING, WAITING, TrackModel, track_key
from partymix.gui.widgets import Card, CurveWidget, DropZone, Legend, TrackTable

ASSETS = Path(__file__).resolve().parent / "assets"
FORMATS = [("FLAC 24-bit (lossless)", "flac"), ("WAV 24-bit", "wav")]
FILE_FILTER = "Audio ({})".format(" ".join("*" + e for e in AUDIO_EXTS))


def app_icon() -> QIcon:
    icon = QIcon()
    for name in ("icon.ico", "icon.png"):
        f = ASSETS / name
        if f.exists():
            icon.addFile(str(f))
    return icon


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().rstrip(".")
    return name or "PartyMix"


def _label(text: str, obj: str = "", size: float = 10, weight=QFont.Weight.Normal, display=False) -> QLabel:
    lab = QLabel(text)
    if obj:
        lab.setObjectName(obj)
    lab.setFont(theme.font(size, weight, display))
    return lab


def _button(text: str, obj: str = "", tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if obj:
        b.setObjectName(obj)
    b.setFont(theme.font(9.5, QFont.Weight.DemiBold if obj in ("Primary", "Hero") else QFont.Weight.Medium))
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    return b


class MainWindow(QMainWindow):
    def __init__(self, backend: Backend, settings: Optional[QSettings] = None, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.settings = settings or QSettings("PartyMix", "PartyMix")
        self.model = TrackModel(self)
        self.model.scorer = backend.pair_score
        self.mixing = False
        self.model_downloading = False
        self.plan = None
        self.plan_seconds: Optional[float] = None
        self.plan_signature = None
        self._plan_req = 0
        self._plan_pending_sig = None
        self._ordering = False
        self._order_touched = False
        self._auto_ordered_once = False
        self.last_outputs: List[str] = []
        self.last_tracklist = ""
        # injectable for tests
        self.confirm = self._confirm_dialog
        self.open_url = QDesktopServices.openUrl

        self.setWindowTitle("PartyMix")
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.setMinimumSize(1100, 720)
        self._build()
        self._connect()
        self._load_settings()
        self._refresh()
        QTimer.singleShot(0, self._check_model)

    # =========================================================================================
    # construction
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 16, 10, 14)
        outer.setSpacing(14)
        outer.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(self._build_tracks_card(), 1)
        left.addWidget(self._build_order_card(), 0)
        body.addLayout(left, 1)
        right_w = QWidget()
        right_w.setObjectName("Root")
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 12, 0)      # 12 px gutter for the (rare) scrollbar
        right.setSpacing(14)
        right.addWidget(self._build_settings_card())
        right.addWidget(self._build_mix_card(), 1)
        # the right column scrolls on small screens instead of squeezing the tracklist
        scroll = QScrollArea()
        scroll.setWidget(right_w)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(360 + 12)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        body.addWidget(scroll)
        outer.addLayout(body, 1)

        self.status = _label("", "Faint", 9)
        self.status.setMinimumHeight(16)
        outer.addWidget(self.status)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self.status.setText(""))

    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        row.setContentsMargins(0, 0, 12, 0)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(40, 40))
        logo.setFixedSize(40, 40)
        row.addWidget(logo)
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(_label("PartyMix", "Brand", 18, QFont.Weight.Bold, display=True))
        col.addWidget(_label("Van losse nummers naar één doorlopende, beat-gematchte mix", "Tag", 9.5))
        row.addLayout(col)
        row.addStretch(1)
        # first-run model download banner
        self.model_box = QFrame()
        self.model_box.setObjectName("Card")
        mb = QHBoxLayout(self.model_box)
        mb.setContentsMargins(14, 8, 14, 8)
        mb.setSpacing(10)
        mcol = QVBoxLayout()
        mcol.setSpacing(4)
        self.model_label = _label("Stemmenmodel downloaden (eenmalig)…", "Muted", 9)
        self.model_bar = QProgressBar()
        self.model_bar.setRange(0, 1000)
        self.model_bar.setFixedHeight(6)
        self.model_bar.setFixedWidth(260)
        mcol.addWidget(self.model_label)
        mcol.addWidget(self.model_bar)
        mb.addLayout(mcol)
        self.model_retry = _button("Opnieuw", "", "Download opnieuw proberen")
        self.model_retry.hide()
        mb.addWidget(self.model_retry)
        self.model_box.hide()
        row.addWidget(self.model_box)
        return row

    def _build_tracks_card(self) -> Card:
        card = Card("Nummers", step=1)
        self.count_chip = _label("", "Chip", 8.5)
        card.header.insertWidget(card.header.count() - 1, self.count_chip)
        self.btn_add = _button("＋  Nummers toevoegen", "Primary")
        self.btn_remove = _button("Verwijderen", "Ghost", "Geselecteerde nummers verwijderen (Delete)")
        self.btn_clear = _button("Lijst leegmaken", "Ghost")
        for b in (self.btn_remove, self.btn_clear, self.btn_add):
            card.header.addWidget(b)

        self.stack = QStackedWidget()
        self.drop_big = DropZone()
        self.drop_big.set_compact(False)
        self.stack.addWidget(self.drop_big)
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(10)
        self.table = TrackTable()
        self.table.setModel(self.model)
        self.table.setup_columns()
        pl.addWidget(self.table, 1)
        self.drop_small = DropZone()
        self.drop_small.set_compact(True)
        pl.addWidget(self.drop_small)
        self.stack.addWidget(page)
        card.lay.addWidget(self.stack, 1)
        return card

    def _build_order_card(self) -> Card:
        card = Card("Volgorde & energie", step=2)
        self.btn_auto = _button("✦  Automatische volgorde", "Primary",
                                "Zet de nummers in de beste volgorde (tempo, toonsoort en energieopbouw)")
        self.btn_first = _button("Vast als eerste", "", "Geselecteerd nummer altijd als opener")
        self.btn_last = _button("Vast als laatste", "", "Geselecteerd nummer altijd als afsluiter")
        for b in (self.btn_first, self.btn_last, self.btn_auto):
            card.header.addWidget(b)
        body = QHBoxLayout()
        body.setSpacing(18)
        cv = QVBoxLayout()
        cv.setSpacing(6)
        self.curve = CurveWidget()
        self.curve.setFixedHeight(168)
        cv.addWidget(self.curve)
        lg = QHBoxLayout()
        lg.addWidget(Legend())
        lg.addStretch(1)
        self.order_hint = _label("Tip: sleep rijen om zelf de volgorde te bepalen · Alt+↑/↓ verplaatst", "Faint", 8.5)
        lg.addWidget(self.order_hint)
        cv.addLayout(lg)
        body.addLayout(cv, 1)
        # stats column
        stats = QFrame()
        stats.setFixedWidth(170)
        sl = QVBoxLayout(stats)
        sl.setContentsMargins(0, 4, 0, 0)
        sl.setSpacing(2)
        sl.addWidget(_label("GESCHATTE LENGTE", "Faint", 8, QFont.Weight.DemiBold))
        self.len_label = _label("–", "Big", 26, QFont.Weight.Bold, display=True)
        sl.addWidget(self.len_label)
        self.len_sub = _label("", "Muted", 9)
        self.len_sub.setWordWrap(True)
        sl.addWidget(self.len_sub)
        sl.addSpacing(10)
        sl.addWidget(_label("TEMPO", "Faint", 8, QFont.Weight.DemiBold))
        self.bpm_label = _label("–", "Big", 13, QFont.Weight.DemiBold, display=True)
        sl.addWidget(self.bpm_label)
        sl.addStretch(1)
        body.addWidget(stats)
        card.lay.addLayout(body)
        return card

    def _build_settings_card(self) -> Card:
        card = Card("Instellingen", step=3)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)
        self.minutes = QSpinBox()
        self.minutes.setRange(5, 240)
        self.minutes.setSingleStep(5)
        self.minutes.setValue(30)
        self.minutes.setSuffix(" minuten")
        self.minutes.lineEdit().setTextMargins(6, 0, 0, 0)
        self.fmt = QComboBox()
        for label, code in FORMATS:
            self.fmt.addItem(label, code)
        self.mp3 = QCheckBox("Ook een MP3 320 kbps maken")
        self.outdir = QLineEdit()
        self.outdir.setPlaceholderText("Kies een map…")
        self.btn_browse = _button("Bladeren…")
        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        dir_row.addWidget(self.outdir, 1)
        dir_row.addWidget(self.btn_browse)
        self.fname = QLineEdit()
        self.fname.setPlaceholderText("PartyMix")
        self.ext_label = _label(".flac", "Faint", 9.5)
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(self.fname, 1)
        name_row.addWidget(self.ext_label)
        rows = [("Doelduur", self.minutes), ("Formaat", self.fmt), ("", self.mp3),
                ("Uitvoermap", dir_row), ("Bestandsnaam", name_row)]
        for r, (text, w) in enumerate(rows):
            if text:
                g.addWidget(_label(text, "FieldLabel", 9.5), r, 0)
            if isinstance(w, QWidget):
                g.addWidget(w, r, 1)
            else:
                g.addLayout(w, r, 1)
        for w in (self.minutes, self.fmt, self.outdir, self.fname, self.mp3):
            w.setFont(theme.font(9.5))
        card.lay.addLayout(g)
        return card

    def _build_mix_card(self) -> Card:
        card = Card("Mix maken", step=4)
        self.btn_mix = _button("Mix maken", "Hero")
        self.btn_mix.setFont(theme.font(12, QFont.Weight.Bold, display=True))
        self.btn_mix.setMinimumHeight(50)
        card.lay.addWidget(self.btn_mix)
        self.mix_hint = _label("", "Muted", 9)
        self.mix_hint.setWordWrap(True)
        self.mix_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        card.lay.addWidget(self.mix_hint)

        # progress
        self.prog_box = QWidget()
        pb = QVBoxLayout(self.prog_box)
        pb.setContentsMargins(0, 4, 0, 0)
        pb.setSpacing(8)
        top = QHBoxLayout()
        self.prog_step = _label("Voorbereiden…", "", 10, QFont.Weight.DemiBold)
        self.prog_pct = _label("0%", "Muted", 10, QFont.Weight.DemiBold)
        self.prog_pct.setMinimumWidth(48)
        self.prog_pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.prog_step, 1)
        top.addWidget(self.prog_pct)
        pb.addLayout(top)
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 1000)
        self.prog_bar.setFixedHeight(10)
        pb.addWidget(self.prog_bar)
        self.prog_sub = _label("Dit kan een paar minuten duren. Je kunt PartyMix gewoon open laten staan.", "Faint", 8.5)
        self.prog_sub.setWordWrap(True)
        pb.addWidget(self.prog_sub)
        self.btn_cancel = _button("Annuleren", "Danger")
        pb.addWidget(self.btn_cancel, 0, Qt.AlignmentFlag.AlignLeft)
        self.prog_box.hide()
        card.lay.addWidget(self.prog_box)

        # result
        self.result_box = QWidget()
        rb = QVBoxLayout(self.result_box)
        rb.setContentsMargins(0, 4, 0, 0)
        rb.setSpacing(8)
        self.result_title = _label("✓  Je mix is klaar!", "", 12, QFont.Weight.Bold, display=True)
        self.result_title.setStyleSheet(f"color: {theme.GREEN};")
        rb.addWidget(self.result_title)
        self.result_file = _label("", "Muted", 9)
        self.result_file.setWordWrap(True)
        self.result_file.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rb.addWidget(self.result_file)
        btns = QHBoxLayout()
        btns.setSpacing(8)
        self.btn_play = _button("▶  Afspelen", "Primary")
        self.btn_folder = _button("Open map")
        self.btn_copy = _button("Kopieer", "Ghost", "Tracklijst naar het klembord kopiëren")
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_folder)
        btns.addStretch(1)
        btns.addWidget(self.btn_copy)
        rb.addLayout(btns)
        rb.addSpacing(4)
        rb.addWidget(_label("TRACKLIJST", "Faint", 8, QFont.Weight.DemiBold))
        self.tracklist = QTableWidget(0, 3)
        self.tracklist.setObjectName("Tracklist")
        self.tracklist.horizontalHeader().hide()
        self.tracklist.verticalHeader().hide()
        self.tracklist.verticalHeader().setDefaultSectionSize(26)
        self.tracklist.setShowGrid(False)
        self.tracklist.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tracklist.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.tracklist.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.tracklist.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tracklist.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tracklist.setWordWrap(False)
        th = self.tracklist.horizontalHeader()
        th.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        th.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        th.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tracklist.setColumnWidth(0, 50)
        self.tracklist.setColumnWidth(1, 26)
        self.tracklist.setMinimumHeight(170)
        rb.addWidget(self.tracklist, 1)
        self.result_box.hide()
        card.lay.addWidget(self.result_box, 1)

        # idle explanation (fills the column before the first mix)
        self.howto = QWidget()
        hl = QVBoxLayout(self.howto)
        hl.setContentsMargins(0, 6, 0, 0)
        hl.setSpacing(9)
        hl.addWidget(_label("ZO WERKT HET", "Faint", 8, QFont.Weight.DemiBold))
        steps = [
            (theme.CYAN, "Analyse", "Beats, tempo, toonsoort, energie en zang van elk nummer."),
            (theme.VIOLET, "Volgorde", "Tempo en toonsoort sluiten aan; de energie bouwt op naar een piek."),
            (theme.PINK, "Overgangen", "Op frasegrenzen, met bas- en zangwissel: zang botst nooit."),
            (theme.GREEN, "Master", "Gelijk volume, −9,5 LUFS, klaar om te draaien."),
        ]
        for color, title, text in steps:
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = QLabel()
            dot.setFixedSize(8, 8)
            dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
            row.addWidget(dot, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            col = QVBoxLayout()
            col.setSpacing(1)
            col.addWidget(_label(title, "", 9.5, QFont.Weight.DemiBold))
            t = _label(text, "Muted", 9)
            t.setWordWrap(True)
            col.addWidget(t)
            row.addLayout(col, 1)
            hl.addLayout(row)
        card.lay.addWidget(self.howto)
        card.lay.addStretch(1)
        return card

    def _connect(self) -> None:
        b = self.backend
        b.analysisStarted.connect(self._on_started)
        b.analysisProgress.connect(self._on_progress)
        b.analysisDone.connect(self._on_done)
        b.analysisFailed.connect(self._on_failed)
        b.orderDone.connect(self._on_order)
        b.orderFailed.connect(self._on_order_failed)
        b.planDone.connect(self._on_plan)
        b.planFailed.connect(self._on_plan_failed)
        b.mixProgress.connect(self._on_mix_progress)
        b.mixDone.connect(self._on_mix_done)
        b.mixFailed.connect(self._on_mix_failed)
        b.mixCancelled.connect(self._on_mix_cancelled)
        b.modelProgress.connect(self._on_model_progress)
        b.modelDone.connect(self._on_model_done)
        b.modelFailed.connect(self._on_model_failed)

        self.btn_add.clicked.connect(self.browse_files)
        self.drop_big.clicked.connect(self.browse_files)
        self.drop_small.clicked.connect(self.browse_files)
        self.drop_big.filesDropped.connect(self.add_files)
        self.drop_small.filesDropped.connect(self.add_files)
        self.table.filesDropped.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_tracks)
        self.btn_auto.clicked.connect(self.auto_order)
        self.btn_first.clicked.connect(lambda: self.pin_selected("first"))
        self.btn_last.clicked.connect(lambda: self.pin_selected("last"))
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._refresh_buttons())
        self.model.orderChanged.connect(self._on_order_changed)
        self.model.layoutChanged.connect(lambda *_: self._mark_touched())
        self.minutes.valueChanged.connect(lambda *_: (self._save_settings(), self._schedule_plan()))
        self.fmt.currentIndexChanged.connect(lambda *_: (self._save_settings(), self._refresh()))
        self.mp3.toggled.connect(lambda *_: self._save_settings())
        self.outdir.editingFinished.connect(self._save_settings)
        self.fname.editingFinished.connect(self._save_settings)
        self.btn_browse.clicked.connect(self.browse_outdir)
        self.btn_mix.clicked.connect(self.start_mix)
        self.btn_cancel.clicked.connect(self.cancel_mix)
        self.btn_play.clicked.connect(self.play_result)
        self.btn_folder.clicked.connect(self.open_result_folder)
        self.btn_copy.clicked.connect(self.copy_tracklist)
        self.model_retry.clicked.connect(self._start_model_download)

        self._plan_timer = QTimer(self)
        self._plan_timer.setSingleShot(True)
        self._plan_timer.setInterval(450)
        self._plan_timer.timeout.connect(self._request_plan)

        QShortcut(QKeySequence.StandardKey.Delete, self.table, activated=self.remove_selected)
        QShortcut(QKeySequence("Alt+Up"), self.table, activated=lambda: self.move_selected(-1))
        QShortcut(QKeySequence("Alt+Down"), self.table, activated=lambda: self.move_selected(1))
        QShortcut(QKeySequence.StandardKey.Open, self, activated=self.browse_files)

    # =========================================================================================
    # settings
    def _load_settings(self) -> None:
        s = self.settings
        music = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MusicLocation) or str(Path.home())
        self.minutes.setValue(int(s.value("minutes", 30)))
        code = s.value("format", "flac")
        self.fmt.setCurrentIndex(max(0, [c for _, c in FORMATS].index(code) if code in [c for _, c in FORMATS] else 0))
        self.mp3.setChecked(str(s.value("mp3", "false")).lower() in ("true", "1"))
        self.outdir.setText(s.value("outdir", music))
        self.outdir.setCursorPosition(0)
        self.fname.setText(s.value("filename", "PartyMix"))
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1360, 860)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("minutes", self.minutes.value())
        s.setValue("format", self.fmt.currentData())
        s.setValue("mp3", self.mp3.isChecked())
        s.setValue("outdir", self.outdir.text())
        s.setValue("filename", self.fname.text())

    def closeEvent(self, e):
        if self.mixing and not self.confirm("Mix wordt gemaakt", "De mix is nog niet klaar. Toch afsluiten?"):
            e.ignore()
            return
        self._plan_timer.stop()
        self._save_settings()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.sync()
        try:
            self.backend.shutdown()
        except Exception:
            pass
        super().closeEvent(e)

    # =========================================================================================
    # adding / removing
    def browse_files(self) -> None:
        if self.mixing:
            return
        start = self.settings.value("lastAddDir", QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MusicLocation))
        files, _ = QFileDialog.getOpenFileNames(self, "Nummers toevoegen", start or "", FILE_FILTER + ";;Alle bestanden (*)")
        if files:
            self.settings.setValue("lastAddDir", str(Path(files[0]).parent))
            self.add_files(files)

    def add_files(self, paths) -> int:
        """Add files and/or folders; returns the number of tracks added."""
        if self.mixing:
            self.flash("Wacht tot de mix klaar is om nummers toe te voegen.")
            return 0
        expanded = expand_inputs(paths)
        audio = [p for p in expanded if is_audio(p)]
        skipped = len(expanded) - len(audio)
        new = self.model.add_paths(audio)
        for it in new:
            self.backend.analyze(it.key, it.path)
        if paths and os.path.isdir(paths[0]):
            self.settings.setValue("lastAddDir", paths[0])
        msg = f"{len(new)} nummer(s) toegevoegd"
        if skipped:
            msg += f" · {skipped} bestand(en) overgeslagen (geen audio)"
        dup = len(audio) - len(new)
        if dup:
            msg += f" · {dup} stond(en) er al in"
        if paths:
            self.flash(msg)
        self._refresh()
        return len(new)

    def selected_rows(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()})

    def remove_selected(self) -> None:
        if self.mixing:
            return
        rows = self.selected_rows()
        for it in self.model.remove_rows(rows):
            if it.status in (WAITING, RUNNING):
                self.backend.cancel_analysis(it.key)
        self._refresh()

    def clear_tracks(self) -> None:
        if self.mixing or not self.model.items:
            return
        if not self.confirm("Lijst leegmaken", f"Alle {len(self.model.items)} nummers uit de lijst halen?"):
            return
        for it in self.model.clear():
            if it.status in (WAITING, RUNNING):
                self.backend.cancel_analysis(it.key)
        self._order_touched = False
        self._auto_ordered_once = False
        self._ordering = False
        self.btn_auto.setText("✦  Automatische volgorde")
        self.result_box.hide()
        self._refresh()

    # =========================================================================================
    # ordering
    def pin_selected(self, which: str) -> None:
        rows = self.selected_rows()
        if len(rows) != 1 or self.mixing:
            return
        it = self.model.items[rows[0]]
        self.model.set_pin(rows[0], "" if it.pin == which else which)
        self._select_key(it.key)

    def move_selected(self, delta: int) -> None:
        rows = self.selected_rows()
        if not rows or self.mixing:
            return
        keys = [self.model.items[r].key for r in rows]
        dest = rows[0] - 1 if delta < 0 else rows[-1] + 2
        self.model.move_rows(rows, max(0, dest))
        self._order_touched = True
        for k in keys:
            self._select_key(k, add=True)

    def _select_key(self, key: str, add: bool = False) -> None:
        r = self.model.find(key)
        if r >= 0:
            from PySide6.QtCore import QItemSelectionModel
            flags = QItemSelectionModel.SelectionFlag.Rows | (
                QItemSelectionModel.SelectionFlag.Select if add else QItemSelectionModel.SelectionFlag.ClearAndSelect)
            self.table.selectionModel().select(self.model.index(r, 0), flags)

    def _mark_touched(self) -> None:
        if not self._ordering:
            self._order_touched = True

    def auto_order(self) -> None:
        ready = self.model.ready_items()
        if len(ready) < 2 or self.mixing:
            return
        first = next((i for i, it in enumerate(ready) if it.pin == "first"), None)
        last = next((i for i, it in enumerate(ready) if it.pin == "last"), None)
        self._ordering = True
        self._order_keys = [it.key for it in ready]
        self.btn_auto.setText("Volgorde bepalen…")
        self.backend.order([it.analysis for it in ready], first, last)
        self._refresh(light=True)

    def _on_order(self, idx) -> None:
        keys = getattr(self, "_order_keys", [])
        self.btn_auto.setText("✦  Automatische volgorde")
        try:
            new_keys = [keys[i] for i in idx]
        except (IndexError, TypeError):
            new_keys = keys
        if not self._ordering:
            return                                   # cleared/cancelled meanwhile
        # keep failed / unfinished tracks after the ordered ones
        self.model.apply_permutation(new_keys)
        self._ordering = False
        self._order_touched = False
        self.flash("Volgorde automatisch bepaald – sleep rijen om zelf te schuiven.")
        self._refresh()

    def _on_order_failed(self, msg: str) -> None:
        self._ordering = False
        self.btn_auto.setText("✦  Automatische volgorde")
        self.flash(f"Volgorde bepalen mislukt: {msg}", error=True)
        self._refresh()

    def _context_menu(self, pos) -> None:
        idx = self.table.indexAt(pos)
        if not idx.isValid() or self.mixing:
            return
        if idx.row() not in self.selected_rows():
            self._select_key(self.model.items[idx.row()].key)
        it = self.model.items[idx.row()]
        m = QMenu(self)
        a_first = m.addAction("Vast als eerste" if it.pin != "first" else "Niet meer vast als eerste")
        a_last = m.addAction("Vast als laatste" if it.pin != "last" else "Niet meer vast als laatste")
        m.addSeparator()
        a_up = m.addAction("Omhoog")
        a_down = m.addAction("Omlaag")
        m.addSeparator()
        a_retry = m.addAction("Opnieuw analyseren") if it.status == FAILED else None
        a_show = m.addAction("Toon in map")
        a_del = m.addAction("Verwijderen")
        chosen = m.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is a_first:
            self.pin_selected("first")
        elif chosen is a_last:
            self.pin_selected("last")
        elif chosen is a_up:
            self.move_selected(-1)
        elif chosen is a_down:
            self.move_selected(1)
        elif a_retry is not None and chosen is a_retry:
            self.model.update_item(it.key, status=WAITING, error="", progress=0.0)
            self.backend.analyze(it.key, it.path)
            self._refresh()
        elif chosen is a_show:
            self.open_url(QUrl.fromLocalFile(str(Path(it.path).parent)))
        elif chosen is a_del:
            self.remove_selected()

    # =========================================================================================
    # analysis events
    def _on_started(self, key: str) -> None:
        it = self.model.update_item(key, status=RUNNING)
        if it is not None:
            self._refresh(light=True)

    def _on_progress(self, key: str, frac: float, text: str) -> None:
        r = self.model.find(key)
        if r < 0:
            return
        it = self.model.items[r]
        changes = {"status": RUNNING, "step": text or it.step}
        if frac >= 0:
            changes["progress"] = frac
        self.model.update_item(key, **changes)

    def _on_done(self, key: str, analysis) -> None:
        if self.model.update_item(key, status=READY, progress=1.0, analysis=analysis, error="") is None:
            return
        self._refresh()
        if self._all_analysed() and not self._order_touched and not self._auto_ordered_once \
                and len(self.model.ready_items()) >= 3:
            self._auto_ordered_once = True
            self.auto_order()

    def _on_failed(self, key: str, msg: str) -> None:
        if self.model.update_item(key, status=FAILED, error=msg or "Analyse mislukt") is not None:
            self._refresh()

    def _all_analysed(self) -> bool:
        return all(it.status in (READY, FAILED) for it in self.model.items)

    # =========================================================================================
    # plan / curve
    def _on_order_changed(self) -> None:
        self._refresh()
        self._schedule_plan()

    def _signature(self):
        return (tuple(it.analysis.file_hash for it in self.model.ready_items()), self.minutes.value())

    def _schedule_plan(self) -> None:
        sig = self._signature()
        if len(sig[0]) < 2:
            self.plan, self.plan_seconds, self.plan_signature = None, None, None
            self._refresh_length()
            return
        if sig == self.plan_signature or sig == self._plan_pending_sig:
            return
        self._plan_timer.start()
        self._refresh_length(planning=True)

    def _request_plan(self) -> None:
        sig = self._signature()
        if len(sig[0]) < 2 or self.mixing:
            return
        self._plan_req += 1
        self._plan_pending_sig = sig
        self._plan_sig_by_req = getattr(self, "_plan_sig_by_req", {})
        self._plan_sig_by_req[self._plan_req] = sig
        self.backend.plan(self._plan_req, [it.analysis for it in self.model.ready_items()], float(sig[1]))

    def _on_plan(self, req: int, plan, seconds) -> None:
        sig = getattr(self, "_plan_sig_by_req", {}).pop(req, None)
        if req != self._plan_req:
            return
        self._plan_pending_sig = None
        self.plan, self.plan_signature = plan, sig
        self.plan_seconds = float(seconds) if seconds is not None else None
        self._refresh_length()

    def _on_plan_failed(self, req: int, msg: str) -> None:
        if req != self._plan_req:
            return
        self._plan_pending_sig = None
        self.plan, self.plan_seconds, self.plan_signature = None, None, None
        self._refresh_length(error=msg)

    def _refresh_length(self, planning: bool = False, error: str = "") -> None:
        ready = self.model.ready_items()
        target = self.minutes.value()
        if self.plan_seconds is not None and self.plan_signature == self._signature():
            self.len_label.setText(fmt_time(self.plan_seconds))
            self.len_label.setStyleSheet("")
            diff = self.plan_seconds / 60 - target
            if abs(diff) < 1.5:
                self.len_sub.setText(f"Doel {target} min – precies goed")
            elif diff < 0:
                self.len_sub.setText(f"Doel {target} min – te weinig nummers voor die duur")
            else:
                self.len_sub.setText(f"Doel {target} min")
        elif len(ready) >= 2 and (planning or self._plan_pending_sig is not None):
            self.len_label.setText("--:--")
            self.len_label.setStyleSheet(f"color: {theme.MUTED};")
            self.len_sub.setText("Overgangen berekenen…")
        elif error:
            self.len_label.setText("--:--")
            self.len_label.setStyleSheet(f"color: {theme.MUTED};")
            self.len_sub.setText(error)
        else:
            self.len_label.setText("--:--")
            self.len_label.setStyleSheet(f"color: {theme.MUTED};")
            self.len_sub.setText(f"Doel {target} min")

    # =========================================================================================
    # mix
    def output_path(self) -> str:
        ext = self.fmt.currentData() or "flac"
        folder = self.outdir.text().strip() or str(Path.home())
        return str(Path(folder) / f"{safe_filename(self.fname.text())}.{ext}")

    def render_settings(self):
        from partymix.model import RenderSettings
        fmt = self.fmt.currentData() or "flac"
        formats = [fmt] + (["mp3"] if self.mp3.isChecked() else [])
        return RenderSettings(out_path=self.output_path(), formats=formats, bit_depth=24)

    def can_mix(self) -> tuple:
        ready = self.model.ready_items()
        if self.mixing:
            return False, ""
        if len(self.model.items) == 0:
            return False, "Voeg eerst nummers toe."
        if len(ready) < 2:
            if any(it.status in (WAITING, RUNNING) for it in self.model.items):
                return False, "Nummers worden geanalyseerd…"
            return False, "Minstens 2 geanalyseerde nummers nodig."
        if any(it.status in (WAITING, RUNNING) for it in self.model.items):
            n = sum(1 for it in self.model.items if it.status in (WAITING, RUNNING))
            return False, f"Nog {n} nummer(s) in analyse…"
        if self.model_downloading:
            return False, "Wacht tot het stemmenmodel gedownload is."
        if self._ordering:
            return False, "Volgorde wordt bepaald…"
        failed = sum(1 for it in self.model.items if it.status == FAILED)
        extra = f" ({failed} met fout worden overgeslagen)" if failed else ""
        return True, f"{len(ready)} nummers · {self.fmt.currentText()}{' + MP3' if self.mp3.isChecked() else ''}{extra}"

    def start_mix(self) -> None:
        ok, _ = self.can_mix()
        if not ok:
            return
        settings = self.render_settings()
        folder = Path(settings.out_path).parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.flash(f"Uitvoermap kan niet worden aangemaakt: {folder}", error=True)
            return
        existing = [str(Path(settings.out_path).with_suffix("." + f)) for f in settings.formats]
        existing = [p for p in existing if os.path.exists(p)]
        if existing and not self.confirm("Bestand bestaat al",
                                         f"{Path(existing[0]).name} bestaat al. Overschrijven?"):
            return
        self._save_settings()
        ready = self.model.ready_items()
        plan = self.plan if self.plan_signature == self._signature() else None
        self.mixing = True
        self._mix_analyses = [it.analysis for it in ready]
        self.result_box.hide()
        self.prog_box.show()
        self.prog_step.setText("Voorbereiden…")
        self.prog_pct.setText("0%")
        self.prog_bar.setValue(0)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Annuleren")
        self.backend.mix(self._mix_analyses, float(self.minutes.value()), settings, plan)
        self._refresh()

    def cancel_mix(self) -> None:
        if not self.mixing:
            return
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Annuleren…")
        self.prog_step.setText("Annuleren…")
        self.backend.cancel_mix()

    def _on_mix_progress(self, frac: float, text: str) -> None:
        if frac >= 0:
            self.prog_bar.setValue(int(frac * 1000))
            self.prog_pct.setText(f"{int(frac * 100)}%")
        if text and self.btn_cancel.isEnabled():
            self.prog_step.setText(text)

    def _on_mix_done(self, plan, outputs) -> None:
        self.mixing = False
        self.prog_box.hide()
        self.last_outputs = [str(o) for o in (outputs or [])] or [self.render_settings().out_path]
        analyses = getattr(self, "_mix_analyses", [])
        total = self.plan_seconds if plan is self.plan else None
        try:
            from partymix.render import estimate_duration
            total = float(estimate_duration(plan, analyses))
        except Exception:
            pass
        try:
            self.last_tracklist = tracklist_text(plan, analyses, total)
            entries = tracklist_entries(plan, analyses)
        except Exception:
            self.last_tracklist, entries = "", []
        self._fill_tracklist(entries, total)
        audio = [o for o in self.last_outputs if is_audio(o)] or self.last_outputs
        self.last_outputs = audio + [o for o in self.last_outputs if o not in audio]
        names = " · ".join(Path(o).name for o in audio)
        self.result_file.setText(names + (f"  ({fmt_time(total)})" if total else ""))
        self.result_file.setToolTip("\n".join(self.last_outputs))
        self.result_box.show()
        mix_sig = (tuple(a.file_hash for a in analyses), self.minutes.value())
        if plan is not None and mix_sig == self._signature():
            self.plan, self.plan_signature = plan, mix_sig
            if total:
                self.plan_seconds = total
        self._schedule_plan()
        self.flash("Mix klaar!")
        QApplication.alert(self)
        self._refresh()

    def _fill_tracklist(self, entries, total) -> None:
        hours = bool(total and total >= 3600)
        self.tracklist.setRowCount(len(entries))
        for r, (t, a) in enumerate(entries):
            name = "?" if a is None else (f"{a.artist} – {a.title}" if a.artist else a.title)
            cells = [(fmt_time(t, hours), theme.CYAN, Qt.AlignmentFlag.AlignRight),
                     (f"{r + 1}", theme.MUTED, Qt.AlignmentFlag.AlignRight),
                     (name, theme.TEXT, Qt.AlignmentFlag.AlignLeft)]
            for c, (text, color, align) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                item.setFont(theme.font(9, QFont.Weight.DemiBold if c == 0 else QFont.Weight.Normal))
                if c == 2:
                    item.setToolTip(name)
                self.tracklist.setItem(r, c, item)

    def _on_mix_failed(self, msg: str) -> None:
        self.mixing = False
        self._schedule_plan()
        self.prog_box.hide()
        self.flash(f"Mix maken mislukt: {msg}", error=True)
        self._refresh()
        self.mix_hint.setText(f"Mislukt: {msg}")
        self.mix_hint.setStyleSheet(f"color: {theme.RED};")

    def _on_mix_cancelled(self) -> None:
        self.mixing = False
        self._schedule_plan()
        self.prog_box.hide()
        self.flash("Mix geannuleerd.")
        self._refresh()

    def play_result(self) -> None:
        if self.last_outputs:
            self.open_url(QUrl.fromLocalFile(self.last_outputs[0]))

    def open_result_folder(self) -> None:
        if self.last_outputs:
            self.open_url(QUrl.fromLocalFile(str(Path(self.last_outputs[0]).parent)))

    def copy_tracklist(self) -> None:
        QGuiApplication.clipboard().setText(self.last_tracklist)
        self.flash("Tracklijst gekopieerd.")

    def browse_outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Uitvoermap kiezen", self.outdir.text())
        if d:
            self.outdir.setText(d)
            self.outdir.setCursorPosition(0)
            self._save_settings()

    # =========================================================================================
    # model download
    def _check_model(self) -> None:
        try:
            missing = self.backend.model_missing()
        except Exception:
            missing = False
        if missing:
            self._start_model_download()

    def _start_model_download(self) -> None:
        self.model_downloading = True
        self.model_box.show()
        self.model_retry.hide()
        self.model_bar.show()
        self.model_bar.setValue(0)
        self.model_label.setText("Stemmenmodel downloaden (eenmalig)…")
        self.model_label.setStyleSheet("")
        self.backend.download_model()
        self._refresh()

    def _on_model_progress(self, frac: float, text: str) -> None:
        if frac >= 0:
            self.model_bar.setValue(int(frac * 1000))
            self.model_label.setText(f"Stemmenmodel downloaden (eenmalig)… {int(frac * 100)}%")

    def _on_model_done(self, downloaded: bool) -> None:
        self.model_downloading = False
        self.model_box.hide()
        if downloaded:
            self.flash("Stemmenmodel gedownload.")
        self._refresh()

    def _on_model_failed(self, msg: str) -> None:
        self.model_downloading = False
        self.model_bar.hide()
        self.model_label.setText(f"Download stemmenmodel mislukt: {msg}")
        self.model_label.setStyleSheet(f"color: {theme.RED};")
        self.model_retry.show()
        self._refresh()

    # =========================================================================================
    # misc
    def flash(self, text: str, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {theme.RED};" if error else "")
        self._status_timer.start(6000)

    def _confirm_dialog(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and not self.mixing:
            e.acceptProposedAction()

    def dropEvent(self, e):
        from partymix.gui.widgets import urls_to_paths
        paths = urls_to_paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.add_files(paths)

    def _refresh_buttons(self) -> None:
        rows = self.selected_rows()
        editable = not self.mixing
        self.btn_remove.setEnabled(editable and bool(rows))
        self.btn_clear.setEnabled(editable and bool(self.model.items))
        self.btn_add.setEnabled(editable)
        one = len(rows) == 1
        self.btn_first.setEnabled(editable and one)
        self.btn_last.setEnabled(editable and one)
        if one:
            pin = self.model.items[rows[0]].pin
            self.btn_first.setText("Losmaken (eerste)" if pin == "first" else "Vast als eerste")
            self.btn_last.setText("Losmaken (laatste)" if pin == "last" else "Vast als laatste")
        self.btn_auto.setEnabled(editable and not self._ordering and len(self.model.ready_items()) >= 2)
        self.table.setDragEnabled(editable)

    def _refresh(self, light: bool = False) -> None:
        items = self.model.items
        self.stack.setCurrentIndex(1 if items else 0)
        ready = self.model.ready_items()
        n_fail = sum(1 for it in items if it.status == FAILED)
        n_busy = sum(1 for it in items if it.status in (WAITING, RUNNING))
        if items:
            total = sum(it.analysis.duration for it in ready)
            parts = [f"{len(items)} nummers"]
            if n_busy:
                parts.append(f"{n_busy} in analyse")
            if n_fail:
                parts.append(f"{n_fail} fout")
            if ready:
                parts.append(f"{fmt_time(total)} totaal")
            self.count_chip.setText("  ·  ".join(parts))
            self.count_chip.setFixedWidth(self.count_chip.fontMetrics().horizontalAdvance(self.count_chip.text()) + 26)
            self.count_chip.show()
        else:
            self.count_chip.hide()
        self.btn_remove.setVisible(bool(items))
        self.btn_clear.setVisible(bool(items))
        self.howto.setVisible(self.result_box.isHidden())
        self._refresh_buttons()
        if not light:
            self.curve.set_data([it.analysis.energy for it in ready], [it.analysis.bpm for it in ready],
                                [it.title for it in ready])
            if ready:
                bpms = [it.analysis.bpm for it in ready]
                lo, hi = min(bpms), max(bpms)
                self.bpm_label.setText(f"{lo:.0f}–{hi:.0f} BPM" if round(hi) != round(lo) else f"{lo:.0f} BPM")
            else:
                self.bpm_label.setText("–")
            self._refresh_length()
        ok, hint = self.can_mix()
        self.btn_mix.setEnabled(ok)
        self.btn_mix.setText("Mix wordt gemaakt…" if self.mixing else "Mix maken")
        if not self.mixing:
            self.mix_hint.setText(hint)
            self.mix_hint.setStyleSheet("")
        self.mix_hint.setVisible(not self.mixing and bool(hint) and self.result_box.isHidden())
        for w in (self.minutes, self.fmt, self.mp3, self.outdir, self.fname, self.btn_browse):
            w.setEnabled(not self.mixing)
        self.ext_label.setText("." + (self.fmt.currentData() or "flac"))
