"""Generate the app icon (assets/icon.png + assets/icon.ico) with QPainter.

    QT_QPA_PLATFORM=offscreen python -m partymix.gui.make_icon
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

ASSETS = Path(__file__).resolve().parent / "assets"


def render_icon(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0
    p.scale(s, s)
    # rounded square with diagonal gradient
    bg = QLinearGradient(0, 0, 256, 256)
    bg.setColorAt(0.0, QColor("#ff3d8b"))
    bg.setColorAt(0.55, QColor("#8b5cf6"))
    bg.setColorAt(1.0, QColor("#22b8ee"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
    # soft highlight
    hl = QRadialGradient(QPointF(80, 50), 170)
    hl.setColorAt(0, QColor(255, 255, 255, 70))
    hl.setColorAt(1, QColor(255, 255, 255, 0))
    p.setBrush(hl)
    p.drawRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
    # two overlapping waveform "tracks" crossing over (the mix)
    def wave(y0, amp, phase, color, width):
        path = QPainterPath()
        for i in range(0, 161):
            x = 48 + i
            env = math.sin(math.pi * i / 160)
            y = y0 + amp * env * math.sin(i / 160 * math.pi * 4 + phase)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    wave(128, 46, 0.0, QColor(255, 255, 255, 120), 13)
    wave(128, 46, math.pi, QColor(255, 255, 255, 255), 13)
    # center dot (the beat)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("white"))
    p.drawEllipse(QPointF(128, 128), 14, 14)
    p.end()
    return img


def _png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def write_ico(path: Path, sizes=(16, 24, 32, 48, 64, 128, 256)) -> None:
    """Multi-resolution .ico with PNG-compressed entries (supported since Windows Vista)."""
    images = [_png_bytes(render_icon(sz)) for sz in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, data = b"", b""
    for sz, png in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", sz % 256, sz % 256, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    path.write_bytes(header + entries + data)


CHEVRON = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
           '<path d="{d}" fill="none" stroke="#a5abbc" stroke-width="1.6" stroke-linecap="round" '
           'stroke-linejoin="round"/></svg>\n')


def write_chevrons() -> None:
    (ASSETS / "chevron_down.svg").write_text(CHEVRON.format(d="M1 1l4 4 4-4"), encoding="utf-8")
    (ASSETS / "chevron_up.svg").write_text(CHEVRON.format(d="M1 5l4-4 4 4"), encoding="utf-8")


def main() -> int:
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # needed for QPainter text/AA
    ASSETS.mkdir(parents=True, exist_ok=True)
    render_icon(256).save(str(ASSETS / "icon.png"))
    write_ico(ASSETS / "icon.ico")
    write_chevrons()
    print(f"Icon geschreven naar {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
