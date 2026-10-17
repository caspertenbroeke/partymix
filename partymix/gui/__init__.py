"""PartyMix desktop GUI (PySide6). ``main()`` starts the application."""
from __future__ import annotations

import sys
from typing import Optional


def create_app(argv=None):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from partymix.gui import theme
    app = QApplication.instance()
    if app is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    app.setApplicationName("PartyMix")
    app.setOrganizationName("PartyMix")
    app.setApplicationDisplayName("PartyMix")
    app.setStyle("Fusion")
    app.setPalette(theme.palette())
    app.setFont(theme.font(10))
    app.setStyleSheet(theme.style_sheet())
    return app


def main(demo: bool = False, workers: Optional[int] = None) -> int:
    if sys.platform == "win32":
        try:  # own taskbar icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PartyMix.PartyMix")
        except Exception:
            pass
    app = create_app()
    from partymix.gui.main_window import MainWindow, app_icon
    app.setWindowIcon(app_icon())
    if demo:
        from partymix.gui.demo import FakeBackend, demo_paths
        backend = FakeBackend(delay_ms=400)
    else:
        from partymix.gui.backend import ProcessBackend
        backend = ProcessBackend(workers)
    win = MainWindow(backend)
    win.show()
    if demo:
        win.add_files(demo_paths())
    return app.exec()
