"""Colours, fonts and the Qt style sheet for PartyMix (dark, clean party look)."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette

# --- palette ---------------------------------------------------------------------------------
BG = "#0d0f15"
BG_2 = "#12151d"
CARD = "#161a24"
CARD_2 = "#1c2130"
BORDER = "#252b3a"
BORDER_2 = "#323a4e"
TEXT = "#eceef4"
TEXT_2 = "#a5abbc"
MUTED = "#6e7488"
PINK = "#ff3d8b"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
GREEN = "#34d399"
ORANGE = "#fbbf24"
RED = "#f87171"

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
DISPLAY_FAMILIES = ["Segoe UI Variable Display", "Segoe UI Semibold", "Inter Display", "Inter", "Segoe UI",
                    "DejaVu Sans"]


def font(size: float = 10, weight: QFont.Weight = QFont.Weight.Normal, display: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(DISPLAY_FAMILIES if display else FONT_FAMILIES)
    f.setPointSizeF(size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


# --- Camelot wheel colours (as on the familiar harmonic-mixing wheel) ------------------------
def camelot_color(code: str) -> QColor:
    """Colour of a Camelot key ("8A", "10B"…). Unknown keys are grey."""
    try:
        num, letter = int(code[:-1]), code[-1].upper()
        if not 1 <= num <= 12 or letter not in "AB":
            raise ValueError
    except (ValueError, IndexError, TypeError):
        return QColor(MUTED)
    hue = (170 - (num - 1) * 30) % 360
    if letter == "A":       # minor: deeper
        return QColor.fromHsv(hue, 185, 225)
    return QColor.fromHsv(hue, 120, 250)   # major: lighter


def energy_color(e: float) -> QColor:
    """1 (chill, cyan) .. 10 (peak, pink) through violet."""
    t = min(max((e - 1) / 9.0, 0.0), 1.0)
    stops = [(0.0, QColor(CYAN)), (0.5, QColor(VIOLET)), (1.0, QColor(PINK))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return QColor(int(c0.red() + (c1.red() - c0.red()) * k),
                          int(c0.green() + (c1.green() - c0.green()) * k),
                          int(c0.blue() + (c1.blue() - c0.blue()) * k))
    return QColor(PINK)


def score_color(score: float) -> QColor:
    if score >= 0.7:
        return QColor(GREEN)
    if score >= 0.4:
        return QColor(ORANGE)
    return QColor(RED)


def palette() -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG, QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: CARD, QPalette.ColorRole.AlternateBase: CARD_2,
        QPalette.ColorRole.Text: TEXT, QPalette.ColorRole.Button: CARD_2,
        QPalette.ColorRole.ButtonText: TEXT, QPalette.ColorRole.Highlight: VIOLET,
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: CARD_2,
        QPalette.ColorRole.ToolTipText: TEXT, QPalette.ColorRole.PlaceholderText: MUTED,
        QPalette.ColorRole.Link: CYAN,
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(MUTED))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(MUTED))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(MUTED))
    return p


_STYLE = f"""
* {{ outline: none; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QWidget {{ color: {TEXT}; }}
QToolTip {{ background: {CARD_2}; color: {TEXT}; border: 1px solid {BORDER_2}; padding: 6px 8px;
            border-radius: 6px; }}

QFrame#Card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel#CardTitle {{ color: {TEXT}; font-weight: 600; }}
QLabel#StepNo {{ color: #ffffff; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {PINK}, stop:1 {VIOLET});
                 border-radius: 11px; font-weight: 700; }}
QLabel#Muted {{ color: {TEXT_2}; }}
QLabel#Faint {{ color: {MUTED}; }}
QLabel#Big {{ color: {TEXT}; }}
QLabel#Brand {{ color: {TEXT}; }}
QLabel#Tag {{ color: {TEXT_2}; }}
QLabel#Chip {{ background: {CARD_2}; border: 1px solid {BORDER_2}; border-radius: 10px; padding: 2px 10px;
               color: {TEXT_2}; }}
QLabel#FieldLabel {{ color: {TEXT_2}; }}

QPushButton {{ background: {CARD_2}; border: 1px solid {BORDER_2}; border-radius: 9px; padding: 7px 14px;
               color: {TEXT}; }}
QPushButton:hover {{ background: #242a3b; border-color: #3d465e; }}
QPushButton:pressed {{ background: #1a1f2c; }}
QPushButton:disabled {{ color: {MUTED}; background: {BG_2}; border-color: {BORDER}; }}
QPushButton#Primary {{ border: none; color: white; font-weight: 600;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {PINK}, stop:1 {VIOLET}); }}
QPushButton#Primary:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ff5c9e, stop:1 #9d74f8); }}
QPushButton#Primary:disabled {{ background: #2a2438; color: #8a7fa3; }}
QPushButton#Hero {{ border: none; color: white; font-weight: 700; border-radius: 12px; padding: 14px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {PINK}, stop:0.55 {VIOLET}, stop:1 #4f7cf7); }}
QPushButton#Hero:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ff5c9e, stop:0.55 #9d74f8, stop:1 #6b92fa); }}
QPushButton#Hero:disabled {{ background: #231f30; color: #7b7392; }}
QPushButton#Ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_2}; }}
QPushButton#Ghost:hover {{ background: {CARD_2}; color: {TEXT}; }}
QPushButton#Ghost:disabled {{ color: #4a5064; background: transparent; }}
QPushButton#Danger {{ background: transparent; border: 1px solid #5b2d3a; color: {RED}; }}
QPushButton#Danger:hover {{ background: #2a1820; }}
QPushButton:checked {{ background: #2b2442; border-color: {VIOLET}; }}

QLineEdit, QSpinBox, QComboBox {{ background: {BG_2}; border: 1px solid {BORDER_2}; border-radius: 8px;
    padding: 6px 9px; color: {TEXT}; selection-background-color: {VIOLET}; min-height: 20px; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {VIOLET}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CARD_2}; border: 1px solid {BORDER_2}; selection-background-color: #2b2442;
    padding: 4px; }}
QSpinBox {{ padding-left: 9px; }}
QSpinBox QLineEdit {{ padding: 0; border: none; background: transparent; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 22px; border: none; background: transparent; }}
QSpinBox::up-arrow {{ image: url(@ASSETS@/chevron_up.svg); width: 10px; height: 6px; }}
QSpinBox::down-arrow {{ image: url(@ASSETS@/chevron_down.svg); width: 10px; height: 6px; }}
QComboBox::down-arrow {{ image: url(@ASSETS@/chevron_down.svg); width: 10px; height: 6px; margin-right: 8px; }}
QCheckBox {{ color: {TEXT}; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {BORDER_2}; background: {BG_2}; }}
QCheckBox::indicator:checked {{ background: {VIOLET}; border-color: {VIOLET}; }}

QTableView {{ background: {CARD}; border: none; gridline-color: transparent; color: {TEXT};
    selection-background-color: #262040; selection-color: {TEXT}; alternate-background-color: #181c27; }}
QTableView::item {{ padding: 0 6px; border: none; }}
QTableView::item:selected {{ background: #262040; }}
QHeaderView {{ background: {CARD}; border: none; }}
QHeaderView::section {{ background: {CARD}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 8px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {CARD}; border: none; }}

QTableWidget#Tracklist {{ background: {BG_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 4px; }}
QTableWidget#Tracklist::item {{ padding: 0 4px; }}

QProgressBar {{ background: {BG_2}; border: none; border-radius: 5px; height: 10px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ border-radius: 5px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {PINK}, stop:1 {CYAN}); }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER_2}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #48516a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_2}; border-radius: 4px; min-width: 30px; }}
QSplitter::handle {{ background: transparent; }}
QMenu {{ background: {CARD_2}; border: 1px solid {BORDER_2}; padding: 5px; border-radius: 8px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: #2b2442; }}
QMessageBox {{ background: {CARD}; }}
"""


def style_sheet() -> str:
    from pathlib import Path
    assets = (Path(__file__).resolve().parent / "assets").as_posix()
    return _STYLE.replace("@ASSETS@", assets)
