"""Custom widgets: cards, drop zone, track table + delegates, energy/BPM curve."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
                           QFontMetrics)
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QStyle, QStyledItemDelegate, QTableView, QVBoxLayout, QWidget)

from partymix.cli import AUDIO_EXTS
from partymix.gui import theme
from partymix.gui.models import (COL_DUR, COL_ENERGY, COL_KEY, COL_NO, COL_PAIR, COL_STATUS, COL_BPM,
                                 COL_TITLE, COL_ARTIST, FAILED, READY, RUNNING, WAITING, ItemRole, PairRole)


def urls_to_paths(mime) -> List[str]:
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()] if mime.hasUrls() else []


# ---------------------------------------------------------------------------------------------
class Card(QFrame):
    """Rounded panel with an optional numbered title row."""

    def __init__(self, title: str = "", step: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(18, 16, 18, 16)
        self.lay.setSpacing(12)
        self.header = QHBoxLayout()
        self.header.setSpacing(10)
        if step is not None:
            no = QLabel(str(step))
            no.setObjectName("StepNo")
            no.setFixedSize(22, 22)
            no.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no.setFont(theme.font(9, QFont.Weight.Bold))
            self.header.addWidget(no)
        if title:
            self.title = QLabel(title)
            self.title.setObjectName("CardTitle")
            self.title.setFont(theme.font(12, QFont.Weight.DemiBold, display=True))
            self.header.addWidget(self.title)
        self.header.addStretch(1)
        if title or step is not None:
            self.lay.addLayout(self.header)


# ---------------------------------------------------------------------------------------------
class DropZone(QFrame):
    """Dashed drop target. ``compact`` = one-line strip shown above a filled table."""
    filesDropped = Signal(list)
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False
        self.compact = False
        self.setMinimumHeight(64)

    def set_compact(self, compact: bool) -> None:
        self.compact = compact
        self.setMinimumHeight(52 if compact else 260)
        self.setMaximumHeight(52 if compact else 16777215)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self.update()

    def dragLeaveEvent(self, e):
        self._hover = False
        self.update()

    def dropEvent(self, e):
        self._hover = False
        self.update()
        paths = urls_to_paths(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        bg = QColor("#1a1530") if self._hover else QColor(theme.BG_2)
        p.setBrush(bg)
        pen = QPen(QColor(theme.VIOLET if self._hover else theme.BORDER_2), 1.6, Qt.PenStyle.DashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.drawRoundedRect(r, 12, 12)
        if self.compact:
            p.setPen(QColor(theme.TEXT_2))
            p.setFont(theme.font(10))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "＋  Sleep meer nummers of een map hierheen – of klik om te bladeren")
            return
        c = r.center()
        # icon: gradient disc with a note-ish glyph
        rad = 34
        g = QLinearGradient(c.x() - rad, c.y() - 70 - rad, c.x() + rad, c.y() - 70 + rad)
        g.setColorAt(0, QColor(theme.PINK))
        g.setColorAt(1, QColor(theme.VIOLET))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(c.x(), c.y() - 58), rad, rad)
        p.setPen(QPen(QColor("white"), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        cx, cy = c.x(), c.y() - 58
        p.drawLine(QPointF(cx, cy - 14), QPointF(cx, cy + 12))
        p.drawLine(QPointF(cx - 12, cy), QPointF(cx, cy - 14))
        p.drawLine(QPointF(cx + 12, cy), QPointF(cx, cy - 14))
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.font(15, QFont.Weight.DemiBold, display=True))
        p.drawText(QRectF(r.left(), c.y() - 8, r.width(), 30), Qt.AlignmentFlag.AlignCenter,
                   "Sleep je nummers hierheen")
        p.setPen(QColor(theme.TEXT_2))
        p.setFont(theme.font(10))
        p.drawText(QRectF(r.left(), c.y() + 24, r.width(), 22), Qt.AlignmentFlag.AlignCenter,
                   "10 tot 30 nummers werkt het best – losse bestanden of een hele map")
        p.setPen(QColor(theme.MUTED))
        p.setFont(theme.font(9))
        exts = " · ".join(e.lstrip(".").upper() for e in AUDIO_EXTS if e != ".aif")
        p.drawText(QRectF(r.left(), c.y() + 50, r.width(), 20), Qt.AlignmentFlag.AlignCenter, exts)


# ---------------------------------------------------------------------------------------------
class TrackDelegate(QStyledItemDelegate):
    """Paints the special columns: number+pin, Camelot pill, energy bar, pair badge, status."""

    def paint(self, p: QPainter, opt, index):
        it = index.data(ItemRole)
        col = index.column()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(opt.rect)
        # background (selection / hover)
        if opt.state & QStyle.StateFlag.State_Selected:
            p.fillRect(opt.rect, QColor("#251f3d"))
            if col == 0:
                p.fillRect(QRectF(r.left(), r.top(), 3, r.height()), QColor(theme.VIOLET))
        elif opt.state & QStyle.StateFlag.State_MouseOver:
            p.fillRect(opt.rect, QColor("#1b2030"))
        # row separator
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(r.bottomLeft() + QPointF(0, -0.5), r.bottomRight() + QPointF(0, -0.5))
        dim = it.status == FAILED
        a = it.analysis
        if col == COL_NO:
            self._number(p, r, index.row() + 1, it.pin)
        elif col == COL_KEY:
            if a is not None and a.key:
                self._pill(p, r, a.key, theme.camelot_color(a.key))
        elif col == COL_ENERGY:
            if a is not None:
                self._energy(p, r, a.energy)
        elif col == COL_PAIR:
            self._pair(p, r, index.data(PairRole))
        elif col == COL_STATUS:
            self._status(p, r, it)
        else:
            text = index.data(Qt.ItemDataRole.DisplayRole) or ""
            if col == COL_TITLE:
                p.setFont(theme.font(10, QFont.Weight.DemiBold))
                p.setPen(QColor(theme.MUTED if dim else theme.TEXT))
            elif col in (COL_BPM, COL_DUR):
                p.setFont(theme.font(10))
                p.setPen(QColor(theme.TEXT_2))
            else:
                p.setFont(theme.font(10))
                p.setPen(QColor(theme.MUTED if dim else theme.TEXT_2))
            align = index.data(Qt.ItemDataRole.TextAlignmentRole) or int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            tr = r.adjusted(10, 0, -10, 0)
            el = QFontMetrics(p.font()).elidedText(text, Qt.TextElideMode.ElideRight, int(tr.width()))
            p.drawText(tr, Qt.AlignmentFlag(align), el)
        p.restore()

    def _number(self, p, r, n, pin):
        p.setFont(theme.font(9.5, QFont.Weight.DemiBold))
        if pin:
            g = QLinearGradient(r.topLeft(), r.bottomRight())
            g.setColorAt(0, QColor(theme.PINK))
            g.setColorAt(1, QColor(theme.VIOLET))
            box = QRectF(0, 0, 26, 20)
            box.moveCenter(r.center())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, str(n))
        else:
            p.setPen(QColor(theme.MUTED))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(n))

    def _pill(self, p, r, text, color: QColor):
        p.setFont(theme.font(9.5, QFont.Weight.Bold))
        w = max(42, QFontMetrics(p.font()).horizontalAdvance(text) + 18)
        box = QRectF(0, 0, w, 22)
        box.moveCenter(r.center())
        bg = QColor(color)
        bg.setAlpha(46)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 150), 1))
        p.setBrush(bg)
        p.drawRoundedRect(box, 11, 11)
        p.setPen(color.lighter(115))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _energy(self, p, r, e):
        n = 10
        seg_w, gap, h = 7, 2.5, 12
        total = n * seg_w + (n - 1) * gap
        x0 = r.left() + 10
        y = r.center().y() - h / 2
        lit = int(round(min(max(e, 1), 10)))
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            rr = QRectF(x0 + i * (seg_w + gap), y, seg_w, h)
            if i < lit:
                p.setBrush(theme.energy_color(1 + i))
            else:
                p.setBrush(QColor("#262c3b"))
            p.drawRoundedRect(rr, 2, 2)
        p.setPen(QColor(theme.TEXT_2))
        p.setFont(theme.font(9))
        p.drawText(QRectF(x0 + total + 8, r.top(), 30, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{e:.1f}")

    def _pair(self, p, r, pair):
        if pair is None:
            return
        score, _reason = pair
        c = theme.score_color(score)
        label = "Top" if score >= 0.7 else ("Matig" if score >= 0.4 else "Lastig")
        p.setFont(theme.font(9, QFont.Weight.DemiBold))
        w = QFontMetrics(p.font()).horizontalAdvance(label) + 30
        box = QRectF(0, 0, w, 22)
        box.moveCenter(r.center())
        bg = QColor(c)
        bg.setAlpha(34)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(box, 11, 11)
        p.setBrush(c)
        p.drawEllipse(QPointF(box.left() + 11, box.center().y()), 3.5, 3.5)
        p.setPen(c)
        p.drawText(box.adjusted(19, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)

    def _status(self, p, r, it):
        p.setFont(theme.font(9.5))
        tr = r.adjusted(10, 0, -10, 0)
        if it.status == RUNNING:
            bar = QRectF(tr.left(), r.center().y() + 5, min(tr.width(), 120), 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#262c3b"))
            p.drawRoundedRect(bar, 2, 2)
            g = QLinearGradient(bar.topLeft(), bar.topRight())
            g.setColorAt(0, QColor(theme.PINK))
            g.setColorAt(1, QColor(theme.CYAN))
            p.setBrush(g)
            done = QRectF(bar.left(), bar.top(), max(bar.width() * it.progress, 4), bar.height())
            p.drawRoundedRect(done, 2, 2)
            p.setPen(QColor(theme.TEXT))
            label = f"{int(it.progress * 100)}%  ·  {it.step or 'Analyseren'}"
            el = QFontMetrics(p.font()).elidedText(label, Qt.TextElideMode.ElideRight, int(tr.width()))
            p.drawText(QRectF(tr.left(), r.top() + 4, tr.width(), r.height() / 2),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, el)
            return
        dot_c, text = {
            WAITING: (theme.MUTED, "Wachten…"),
            READY: (theme.GREEN, "Klaar"),
            FAILED: (theme.RED, "Fout – " + (it.error or "onbekend")),
        }.get(it.status, (theme.MUTED, it.status))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(dot_c))
        p.drawEllipse(QPointF(tr.left() + 4, r.center().y()), 3.5, 3.5)
        p.setPen(QColor(theme.RED if it.status == FAILED else theme.TEXT_2))
        el = QFontMetrics(p.font()).elidedText(text, Qt.TextElideMode.ElideRight, int(tr.width() - 14))
        p.drawText(tr.adjusted(14, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, el)

    def sizeHint(self, opt, index):
        return QSize(80, 40)


class TrackTable(QTableView):
    """Table with internal drag-to-reorder and external file drops."""
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(TrackDelegate(self))
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setShowGrid(False)
        self.setMouseTracking(True)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(40)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        h = self.horizontalHeader()
        h.setHighlightSections(False)
        h.setSectionsClickable(False)
        h.setFont(theme.font(8.5, QFont.Weight.DemiBold))

    FIXED = {COL_NO: 42, COL_BPM: 62, COL_KEY: 84, COL_ENERGY: 122, COL_DUR: 56, COL_PAIR: 88, COL_STATUS: 150}

    def setup_columns(self):
        h = self.horizontalHeader()
        for c, w in self.FIXED.items():
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(c, w)
        h.setSectionResizeMode(COL_TITLE, QHeaderView.ResizeMode.Fixed)
        h.setSectionResizeMode(COL_ARTIST, QHeaderView.ResizeMode.Fixed)
        self._fit_columns()

    def _fit_columns(self):
        """Title gets 60 % and artist 40 % of the space left by the fixed columns."""
        free = max(self.viewport().width() - sum(self.FIXED.values()), 160)
        self.setColumnWidth(COL_TITLE, int(free * 0.6))
        self.setColumnWidth(COL_ARTIST, free - int(free * 0.6))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_columns()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = urls_to_paths(e.mimeData())
            e.acceptProposedAction()
            if paths:
                self.filesDropped.emit(paths)
            return
        super().dropEvent(e)


# ---------------------------------------------------------------------------------------------
def target_energy(x: float) -> float:
    """Indicative energy shape 0..1 → 1..10: warm-up, build, peak at ~80 %, short finale."""
    if x < 0.8:
        y = 4.5 + 4.7 * (x / 0.8) ** 1.3
    else:
        y = 9.2 - 1.6 * ((x - 0.8) / 0.2) ** 2
    return y


class CurveWidget(QWidget):
    """Energy (area) and BPM (line) of the running order, with the target curve dashed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.energies: List[float] = []
        self.bpms: List[float] = []
        self.labels: List[str] = []
        self.setMinimumHeight(170)

    def set_data(self, energies: Sequence[float], bpms: Sequence[float], labels: Sequence[str] = ()):
        self.energies, self.bpms, self.labels = list(energies), list(bpms), list(labels)
        self.update()

    @staticmethod
    def _smooth(pts: List[QPointF]) -> QPainterPath:
        path = QPainterPath(pts[0])
        for i in range(1, len(pts)):
            p0 = pts[i - 2] if i >= 2 else pts[i - 1]
            p1, p2 = pts[i - 1], pts[i]
            p3 = pts[i + 1] if i + 1 < len(pts) else p2
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            path.cubicTo(c1, c2, p2)
        return path

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        left, right, top, bottom = 34, 44, 10, 24
        plot = QRectF(left, top, W - left - right, H - top - bottom)
        p.setFont(theme.font(8))
        # grid
        for e in (2, 4, 6, 8, 10):
            y = plot.bottom() - (e - 1) / 9 * plot.height()
            p.setPen(QPen(QColor("#1f2433"), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(e))
        n = len(self.energies)
        if n == 0:
            p.setPen(QColor(theme.MUTED))
            p.setFont(theme.font(10))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "De energiecurve verschijnt zodra nummers zijn geanalyseerd")
            return
        xs = [plot.left() + (plot.width() * (i / (n - 1)) if n > 1 else plot.width() / 2) for i in range(n)]
        # target curve
        tgt = [QPointF(plot.left() + plot.width() * k / 60,
                       plot.bottom() - (target_energy(k / 60) - 1) / 9 * plot.height()) for k in range(61)]
        pen = QPen(QColor(255, 255, 255, 60), 1.3, Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 4])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self._smooth(tgt))
        # energy area
        pts = [QPointF(x, plot.bottom() - (min(max(e, 1), 10) - 1) / 9 * plot.height()) for x, e in zip(xs, self.energies)]
        line = self._smooth(pts) if n > 1 else QPainterPath(pts[0])
        area = QPainterPath(line)
        area.lineTo(xs[-1], plot.bottom())
        area.lineTo(xs[0], plot.bottom())
        area.closeSubpath()
        g = QLinearGradient(0, plot.top(), 0, plot.bottom())
        g.setColorAt(0, QColor(255, 61, 139, 110))
        g.setColorAt(1, QColor(139, 92, 246, 8))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawPath(area)
        lg = QLinearGradient(plot.left(), 0, plot.right(), 0)
        lg.setColorAt(0, QColor(theme.VIOLET))
        lg.setColorAt(1, QColor(theme.PINK))
        p.setPen(QPen(QBrush(lg), 2.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(line)
        # BPM line (right axis)
        if self.bpms:
            lo, hi = min(self.bpms), max(self.bpms)
            pad = max(2.0, (hi - lo) * 0.25)
            lo, hi = math.floor(lo - pad), math.ceil(hi + pad)
            bp = [QPointF(x, plot.bottom() - (b - lo) / (hi - lo) * plot.height()) for x, b in zip(xs, self.bpms)]
            p.setPen(QPen(QColor(34, 211, 238, 200), 1.6))
            for a_, b_ in zip(bp, bp[1:]):
                p.drawLine(a_, b_)
            p.setBrush(QColor(theme.BG))
            for pt in bp:
                p.drawEllipse(pt, 2.6, 2.6)
            p.setPen(QColor(theme.CYAN))
            for v in (lo, (lo + hi) / 2, hi):
                y = plot.bottom() - (v - lo) / (hi - lo) * plot.height()
                p.drawText(QRectF(plot.right() + 6, y - 8, right - 6, 16),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{v:.0f}")
        # points + x labels
        for i, (x, pt) in enumerate(zip(xs, pts)):
            p.setPen(QPen(QColor("white"), 1.5))
            p.setBrush(theme.energy_color(self.energies[i]))
            p.drawEllipse(pt, 4, 4)
            if n <= 20 or i % 2 == 0:
                p.setPen(QColor(theme.MUTED))
                p.drawText(QRectF(x - 14, plot.bottom() + 6, 28, 14), Qt.AlignmentFlag.AlignCenter, str(i + 1))


class Legend(QWidget):
    """Small inline legend for the curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        for color, text, dashed in ((theme.PINK, "Energie", False), (theme.CYAN, "BPM", False),
                                    ("#8a8f9f", "Streefcurve", True)):
            sw = _Swatch(color, dashed)
            lab = QLabel(text)
            lab.setObjectName("Faint")
            lab.setFont(theme.font(8.5))
            row = QHBoxLayout()
            row.setSpacing(5)
            row.addWidget(sw)
            row.addWidget(lab)
            lay.addLayout(row)


class _Swatch(QWidget):
    def __init__(self, color, dashed):
        super().__init__()
        self.color, self.dashed = color, dashed
        self.setFixedSize(16, 10)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(self.color), 2.2, Qt.PenStyle.DashLine if self.dashed else Qt.PenStyle.SolidLine)
        if self.dashed:
            pen.setDashPattern([2, 2])
        p.setPen(pen)
        p.drawLine(1, 5, 15, 5)
