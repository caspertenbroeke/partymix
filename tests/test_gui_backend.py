"""MainWindow + real ProcessBackend (spawned worker processes) with a stand-in analysis job."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings

from partymix.gui import create_app
from partymix.gui.backend import ProcessBackend, normalise_score
from partymix.gui.main_window import MainWindow
from partymix.gui.models import FAILED, READY


def test_normalise_score():
    assert normalise_score(85) == 0.85 and normalise_score(0.3) == 0.3 and normalise_score(-1) == 0


def test_process_backend_analysis_with_crash(qtbot, tmp_path):
    create_app([])
    backend = ProcessBackend(workers=2, job_refs={"analyze": "test_cli_worker:job_analyze_fake"})
    backend.model_missing = lambda: False
    w = MainWindow(backend, QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    w.confirm = lambda *a: True
    qtbot.addWidget(w)
    try:
        w.add_files(["/m/01 - Avicii - Levels.mp3", "/m/crash.mp3", "/m/Die Young.mp3"])
        qtbot.waitUntil(lambda: all(it.status in (READY, FAILED) for it in w.model.items), timeout=60000)
        st = {it.title: it for it in w.model.items}
        assert st["Levels"].status == READY and st["Die Young"].status == READY
        assert st["crash"].status == FAILED and "gestopt" in st["crash"].error
    finally:
        backend.shutdown()
