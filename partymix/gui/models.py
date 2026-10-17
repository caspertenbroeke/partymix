"""Track list model (the table *is* the running order)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QAbstractTableModel, QByteArray, QMimeData, QModelIndex, Qt, Signal

from partymix.cli import fmt_time, guess_title_artist

WAITING, RUNNING, READY, FAILED = "wacht", "bezig", "klaar", "fout"

COLUMNS = ["#", "Titel", "Artiest", "BPM", "Toonsoort", "Energie", "Duur", "Overgang", "Status"]
(COL_NO, COL_TITLE, COL_ARTIST, COL_BPM, COL_KEY, COL_ENERGY, COL_DUR, COL_PAIR, COL_STATUS) = range(len(COLUMNS))

ItemRole = Qt.ItemDataRole.UserRole + 1
PairRole = Qt.ItemDataRole.UserRole + 2
MIME_ROWS = "application/x-partymix-rows"


@dataclass
class TrackItem:
    path: str
    title: str = ""
    artist: str = ""
    status: str = WAITING
    progress: float = 0.0
    step: str = ""
    error: str = ""
    analysis: object = None
    pin: str = ""                      # "" | "first" | "last"

    @property
    def key(self) -> str:
        return track_key(self.path)

    @property
    def ready(self) -> bool:
        return self.status == READY and self.analysis is not None


def track_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class TrackModel(QAbstractTableModel):
    orderChanged = Signal()            # rows moved / added / removed / analyses changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[TrackItem] = []
        self.scorer: Optional[Callable] = None          # (a, b) -> (score 0..1, reason) | None
        self._pair_cache: Dict[Tuple[str, str], Optional[Tuple[float, str]]] = {}

    # --- Qt model API -------------------------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                return COLUMNS[section]
            if role == Qt.ItemDataRole.TextAlignmentRole:
                if section in (COL_BPM, COL_DUR):
                    return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if section in (COL_NO, COL_KEY, COL_PAIR):
                    return int(Qt.AlignmentFlag.AlignCenter)
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        it = self.items[index.row()]
        col = index.column()
        a = it.analysis
        if role == ItemRole:
            return it
        if role == PairRole and col == COL_PAIR:
            return self.pair(index.row())
        if role == Qt.ItemDataRole.DisplayRole:
            if col == COL_NO:
                return str(index.row() + 1)
            if col == COL_TITLE:
                return it.title
            if col == COL_ARTIST:
                return it.artist
            if col == COL_BPM:
                return f"{a.bpm:.1f}" if a is not None else ""
            if col == COL_KEY:
                return a.key if a is not None else ""
            if col == COL_DUR:
                return fmt_time(a.duration) if a is not None else ""
            if col == COL_STATUS:
                return status_text(it)
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (COL_BPM, COL_DUR):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if col in (COL_NO, COL_KEY):
                return int(Qt.AlignmentFlag.AlignCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == COL_PAIR:
                p = self.pair(index.row())
                if p is not None:
                    prev = self.items[index.row() - 1]
                    return f"<b>{prev.title}</b> → <b>{it.title}</b><br>{p[1]}<br>Score: {int(p[0] * 100)}/100"
                return None
            if col == COL_STATUS and it.status == FAILED:
                return it.error
            if col == COL_STATUS and it.status == RUNNING:
                return f"{it.step or 'Analyseren'} – {int(it.progress * 100)}%"
            if col in (COL_TITLE, COL_ARTIST):
                return it.path
            if col == COL_KEY and a is not None and a.key:
                return f"Camelot {a.key}"
            if col == COL_ENERGY and a is not None:
                return f"Energie {a.energy:.1f} / 10"
        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        return base

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction | Qt.DropAction.CopyAction

    def supportedDragActions(self):
        return Qt.DropAction.MoveAction

    def mimeTypes(self):
        return [MIME_ROWS, "text/uri-list"]

    def mimeData(self, indexes):
        rows = sorted({i.row() for i in indexes})
        md = QMimeData()
        md.setData(MIME_ROWS, QByteArray(",".join(map(str, rows)).encode()))
        return md

    def canDropMimeData(self, data, action, row, column, parent):
        return data.hasFormat(MIME_ROWS)

    def dropMimeData(self, data, action, row, column, parent):
        if not data.hasFormat(MIME_ROWS):
            return False
        rows = [int(x) for x in bytes(data.data(MIME_ROWS)).decode().split(",") if x]
        if row < 0:
            row = parent.row() if parent.isValid() else len(self.items)
        self.move_rows(rows, row)
        return False          # we moved the rows ourselves: the view must not delete anything

    # --- editing API --------------------------------------------------------------------------
    def keys(self) -> List[str]:
        return [it.key for it in self.items]

    def find(self, key: str) -> int:
        for i, it in enumerate(self.items):
            if it.key == key:
                return i
        return -1

    def add_paths(self, paths) -> List[TrackItem]:
        existing = set(self.keys())
        new = []
        for p in paths:
            k = track_key(p)
            if k in existing:
                continue
            existing.add(k)
            title, artist = guess_title_artist(p)
            new.append(TrackItem(path=p, title=title, artist=artist))
        if new:
            # keep a pinned last track at the end
            pos = len(self.items)
            if self.items and self.items[-1].pin == "last":
                pos -= 1
            self.beginInsertRows(QModelIndex(), pos, pos + len(new) - 1)
            self.items[pos:pos] = new
            self.endInsertRows()
            self.orderChanged.emit()
        return new

    def remove_rows(self, rows) -> List[TrackItem]:
        removed = []
        for r in sorted(set(rows), reverse=True):
            if 0 <= r < len(self.items):
                self.beginRemoveRows(QModelIndex(), r, r)
                removed.append(self.items.pop(r))
                self.endRemoveRows()
        if removed:
            self.orderChanged.emit()
        return removed

    def clear(self) -> List[TrackItem]:
        old = self.items
        self.beginResetModel()
        self.items = []
        self.endResetModel()
        self.orderChanged.emit()
        return old

    def move_rows(self, rows, dest: int) -> None:
        """Move the given rows (keeping their relative order) so they land before ``dest``."""
        rows = sorted({r for r in rows if 0 <= r < len(self.items)})
        if not rows:
            return
        dest = max(0, min(dest, len(self.items)))
        moving = [self.items[r] for r in rows]
        rest = [it for i, it in enumerate(self.items) if i not in rows]
        insert_at = dest - sum(1 for r in rows if r < dest)
        new = rest[:insert_at] + moving + rest[insert_at:]
        if new == self.items:
            return
        self._apply_order(new, manual=True)

    def apply_permutation(self, keys_in_order: List[str]) -> None:
        """Reorder to the given keys; items not mentioned keep their relative order at the end."""
        by_key = {it.key: it for it in self.items}
        new = [by_key[k] for k in keys_in_order if k in by_key]
        new += [it for it in self.items if it not in new]
        self._apply_order(new, manual=False)

    def _apply_order(self, new: List[TrackItem], manual: bool) -> None:
        if manual:   # a manually moved pinned track loses its pin
            for i, it in enumerate(new):
                if it.pin == "first" and i != 0:
                    it.pin = ""
                if it.pin == "last" and i != len(new) - 1:
                    it.pin = ""
        self.layoutAboutToBeChanged.emit()
        old = self.items
        persistent = self.persistentIndexList()
        self.items = new
        new_rows = {id(it): i for i, it in enumerate(new)}
        for idx in persistent:
            if 0 <= idx.row() < len(old):
                self.changePersistentIndex(idx, self.index(new_rows[id(old[idx.row()])], idx.column()))
        self.layoutChanged.emit()
        self.orderChanged.emit()

    def set_pin(self, row: int, pin: str) -> None:
        if not 0 <= row < len(self.items):
            return
        target = self.items[row]
        for it in self.items:
            if it.pin == pin:
                it.pin = ""
        target.pin = pin
        new = [it for it in self.items if it is not target]
        if pin == "first":
            new.insert(0, target)
        elif pin == "last":
            new.append(target)
        else:
            new = self.items[:]
        self._apply_order(new, manual=False)

    def update_item(self, key: str, **changes) -> Optional[TrackItem]:
        r = self.find(key)
        if r < 0:
            return None
        it = self.items[r]
        for k, v in changes.items():
            setattr(it, k, v)
        if "analysis" in changes:
            a = it.analysis
            if a is not None:
                it.title = a.title or it.title
                it.artist = a.artist or it.artist
            self.dataChanged.emit(self.index(max(r - 1, 0), 0), self.index(min(r + 1, len(self.items) - 1),
                                                                            len(COLUMNS) - 1))
            self.orderChanged.emit()
        else:
            self.dataChanged.emit(self.index(r, 0), self.index(r, len(COLUMNS) - 1))
        return it

    # --- derived ------------------------------------------------------------------------------
    def ready_items(self) -> List[TrackItem]:
        return [it for it in self.items if it.ready]

    def pair(self, row: int) -> Optional[Tuple[float, str]]:
        """Compatibility of the ready track before ``row`` (skipping failed ones) with ``row``."""
        if row <= 0 or row >= len(self.items) or not self.items[row].ready:
            return None
        prev = next((self.items[r] for r in range(row - 1, -1, -1) if self.items[r].ready), None)
        if prev is None or self.scorer is None:
            return None
        k = (prev.analysis.file_hash, self.items[row].analysis.file_hash)
        if k not in self._pair_cache:
            try:
                self._pair_cache[k] = self.scorer(prev.analysis, self.items[row].analysis)
            except Exception:
                self._pair_cache[k] = None
        return self._pair_cache[k]


def status_text(it: TrackItem) -> str:
    if it.status == WAITING:
        return "Wachten…"
    if it.status == RUNNING:
        return f"Analyseren {int(it.progress * 100)}%"
    if it.status == READY:
        return "Klaar"
    return "Fout"
