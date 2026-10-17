"""Headless GUI smoke tests (pytest-qt). Also writes screenshots to test_out/gui_*.png."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QModelIndex, QSettings, QUrl, Qt

from partymix.gui import create_app
from partymix.gui.demo import FakeBackend, analysis_for_path, demo_paths
from partymix.gui.main_window import MainWindow
from partymix.gui.models import COL_PAIR, FAILED, MIME_ROWS, READY, RUNNING, PairRole

OUT = Path(__file__).resolve().parent.parent / "test_out"


@pytest.fixture(scope="session")
def qapp():
    return create_app([])


@pytest.fixture
def make_window(qapp, qtbot, tmp_path):
    def make(backend=None, **kw):
        settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
        settings.setValue("outdir", str(tmp_path / "out"))
        backend = backend or FakeBackend(**kw)
        w = MainWindow(backend, settings)
        w.confirm = lambda *a: True
        w.opened = []
        w.open_url = lambda url: w.opened.append(url.toLocalFile())
        w.resize(1440, 900)
        qtbot.addWidget(w)
        w.show()
        qtbot.waitExposed(w)
        return w
    return make


def shot(w, name):
    from PySide6.QtWidgets import QApplication
    OUT.mkdir(exist_ok=True)
    for _ in range(3):
        QApplication.processEvents()
    w.repaint()
    assert w.grab().save(str(OUT / f"gui_{name}.png"))


def analysed_window(make_window, qtbot, n=14):
    w = make_window()
    w.add_files(demo_paths()[:n])
    qtbot.waitUntil(lambda: len(w.model.ready_items()) == n, timeout=5000)
    return w


def test_window_opens_empty(make_window):
    w = make_window()
    assert w.windowTitle() == "PartyMix"
    assert w.stack.currentIndex() == 0          # big drop zone
    assert not w.btn_mix.isEnabled()
    assert "Voeg eerst" in w.mix_hint.text()
    assert not w.windowIcon().isNull()
    shot(w, "empty")


def test_add_files_fills_table_and_auto_orders(make_window, qtbot):
    w = analysed_window(make_window, qtbot)
    assert w.model.rowCount() == 14
    assert w.stack.currentIndex() == 1
    # first full analysis triggers one automatic ordering (energy ascending in the fake planner)
    qtbot.waitUntil(lambda: any(r[0] == "order" for r in w.backend.requests), timeout=3000)
    qtbot.waitUntil(lambda: not w._ordering, timeout=3000)
    energies = [it.analysis.energy for it in w.model.items]
    assert energies == sorted(energies)
    # plan → estimated length
    qtbot.waitUntil(lambda: w.plan_seconds is not None, timeout=3000)
    assert ":" in w.len_label.text()
    assert w.btn_mix.isEnabled()
    # pair badges + tooltips
    pair = w.model.index(1, COL_PAIR).data(PairRole)
    assert pair is not None and 0 <= pair[0] <= 1
    assert "→" in w.model.index(1, COL_PAIR).data(Qt.ItemDataRole.ToolTipRole)
    assert w.model.index(0, COL_PAIR).data(PairRole) is None
    # duplicates and non-audio are ignored
    assert w.add_files(demo_paths()[:2] + ["/tmp/readme.txt"]) == 0
    w.table.selectRow(5)
    shot(w, "analysed")


def test_failed_track_is_marked_and_others_continue(make_window, qtbot):
    w = make_window()
    paths = demo_paths()[:3] + ["/muziek/kapot bestand.mp3"]
    w.add_files(paths)
    qtbot.waitUntil(lambda: all(it.status in (READY, FAILED) for it in w.model.items), timeout=5000)
    bad = [it for it in w.model.items if it.status == FAILED]
    assert len(bad) == 1 and "mislukt" in bad[0].error
    assert len(w.model.ready_items()) == 3
    ok, hint = w.can_mix()
    assert ok and "overgeslagen" in hint


def test_manual_reorder_pin_and_drop(make_window, qtbot):
    w = analysed_window(make_window, qtbot, n=5)
    qtbot.waitUntil(lambda: not w._ordering, timeout=3000)
    titles = lambda: [it.title for it in w.model.items]
    before = titles()
    # drag & drop through the model's mime API (what the view does)
    md = w.model.mimeData([w.model.index(0, 0)])
    assert md.hasFormat(MIME_ROWS)
    w.model.dropMimeData(md, Qt.DropAction.MoveAction, 3, 0, QModelIndex())
    assert titles() == before[1:3] + [before[0]] + before[3:]
    assert w._order_touched
    # Alt+Down style move
    w.table.selectRow(0)
    w.move_selected(1)
    assert titles()[1] == before[1]
    # pin last → moves to the end and is passed to order_tracks
    w.table.selectRow(1)
    key = w.model.items[1].key
    w.pin_selected("last")
    assert w.model.items[-1].key == key and w.model.items[-1].pin == "last"
    w.table.selectRow(2)
    first_key = w.model.items[2].key
    w.pin_selected("first")
    assert w.model.items[0].key == first_key
    w.auto_order()
    qtbot.waitUntil(lambda: not w._ordering, timeout=3000)
    req = [r for r in w.backend.requests if r[0] == "order"][-1]
    assert req[1] == 0 and req[2] == 4
    assert w.model.items[0].key == first_key and w.model.items[-1].key == key
    # removing works
    w.table.selectRow(2)
    w.remove_selected()
    assert w.model.rowCount() == 4


def test_external_file_drop_on_table(make_window, qtbot):
    w = analysed_window(make_window, qtbot, n=2)
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(p) for p in demo_paths()[2:4]])
    w.table.filesDropped.emit([u.toLocalFile() for u in md.urls()])
    assert w.model.rowCount() == 4


def test_render_flow_with_mocked_worker(make_window, qtbot, tmp_path):
    w = analysed_window(make_window, qtbot)
    qtbot.waitUntil(lambda: w.plan_seconds is not None and not w._ordering, timeout=3000)
    w.mp3.setChecked(True)
    w.fname.setText("Feest: zaterdag?")          # illegal chars are stripped
    w.backend.delay_ms = 150
    w.start_mix()
    assert w.mixing and w.prog_box.isVisible() and not w.btn_mix.isEnabled()
    s = w.backend.mix_settings
    assert s.formats == ["flac", "mp3"] and s.out_path.endswith("Feest zaterdag.flac")
    qtbot.waitUntil(lambda: w.prog_bar.value() >= 300, timeout=3000)
    shot(w, "rendering")
    qtbot.waitUntil(lambda: not w.mixing, timeout=5000)
    assert w.result_box.isVisible()
    assert len(w.last_outputs) == 2
    lines = w.last_tracklist.splitlines()
    assert len(lines) == 14 and lines[0].startswith("0:00")
    w.play_result()
    w.open_result_folder()
    assert w.opened[0].endswith(".flac") and w.opened[1] == str(Path(s.out_path).parent)
    shot(w, "done")
    # settings persisted
    assert w.settings.value("mp3") in (True, "true")


def test_cancel_mix(make_window, qtbot):
    w = analysed_window(make_window, qtbot, n=3)
    qtbot.waitUntil(lambda: w.can_mix()[0], timeout=3000)
    w.backend.delay_ms = 300
    w.start_mix()
    w.cancel_mix()
    qtbot.waitUntil(lambda: not w.mixing, timeout=3000)
    assert not w.result_box.isVisible()
    assert "geannuleerd" in w.status.text().lower()


def test_model_download_first_run(make_window, qtbot):
    w = make_window(model_missing=True, delay_ms=30)
    qtbot.waitUntil(lambda: w.model_box.isVisible(), timeout=2000)
    assert w.model_downloading
    qtbot.waitUntil(lambda: w.model_bar.value() > 0, timeout=2000)
    shot(w, "model_download")
    qtbot.waitUntil(lambda: not w.model_downloading, timeout=3000)
    assert not w.model_box.isVisible()


def test_analysing_state_screenshot(make_window, qtbot):
    w = make_window(auto=False)
    w.add_files(demo_paths())
    b = w.backend
    for i, it in enumerate(w.model.items[:6]):
        b.analysisDone.emit(it.key, analysis_for_path(it.path))
    b.analysisStarted.emit(w.model.items[6].key)
    b.analysisProgress.emit(w.model.items[6].key, 0.64, "Stemmen scheiden")
    b.analysisStarted.emit(w.model.items[7].key)
    b.analysisProgress.emit(w.model.items[7].key, 0.18, "Beats zoeken")
    b.analysisFailed.emit(w.model.items[8].key, "Analyse mislukt: bestand is beschadigd")
    assert w.model.items[6].status == RUNNING
    assert not w.btn_mix.isEnabled() and "analyse" in w.mix_hint.text()
    shot(w, "analysing")
