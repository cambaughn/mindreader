"""Mindreader's look: one cool-dark neutral, one teal accent, red only while recording.

Applied with Qt's Fusion style so the stylesheet is honoured on macOS. Widgets opt into roles with
`setProperty("role", ...)` (muted, hero, mono, chip) or an objectName (primary buttons, the capture pill).
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QWidget

P = {
    "bg": "#14161b",
    "surface": "#1c1f26",
    "surface2": "#242831",
    "border": "#2f343e",
    "text": "#e9ebf0",
    "muted": "#8d94a3",
    "accent": "#5cc8bb",
    "accent_dim": "#2a4a47",
    "rec": "#e5484d",
    "good": "#4cc38a",
    "warn": "#e2a73f",
    "bad": "#ef6a62",
}

# Read-back word colours: right / swapped / missed / extra.
OP_COLORS = {"ok": P["good"], "sub": P["warn"], "del": P["bad"], "ins": P["muted"]}

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {P['bg']}; color: {P['text']}; font-size: 13px; }}
QToolTip {{ background: {P['surface2']}; color: {P['text']}; border: 1px solid {P['border']}; padding: 6px; }}

QLabel[role="muted"] {{ color: {P['muted']}; }}
QLabel[role="eyebrow"] {{ color: {P['muted']}; font-size: 11px; letter-spacing: 1px; }}
QLabel[role="hero"] {{ font-size: 30px; font-weight: 500; line-height: 1.2; }}
QLabel[role="prompt"] {{ font-size: 26px; font-weight: 500; }}
QLabel[role="readback"] {{ font-size: 18px; }}
QLabel[role="mono"] {{ font-family: Menlo, monospace; font-size: 12px; color: {P['muted']}; }}

QLabel[role="chip"] {{ padding: 3px 10px 3px 8px; border-radius: 10px; font-size: 12px;
                       background: {P['surface2']}; color: {P['muted']}; }}
QLabel[role="chip"][ok="true"] {{ color: {P['good']}; }}
QLabel[role="chip"][ok="false"] {{ color: {P['bad']}; }}

QFrame[role="card"] {{ background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 10px; }}
QFrame[role="card"] QLabel {{ background: transparent; }}

QLabel#capture {{ font-size: 15px; font-weight: 500; padding: 10px; border-radius: 8px;
                  border: 1px solid {P['border']}; color: {P['muted']}; background: {P['surface']}; }}
QLabel#capture[recording="true"] {{ background: {P['rec']}; border-color: {P['rec']}; color: white; }}
QLabel#capture[armed="true"] {{ color: {P['text']}; border-color: {P['accent_dim']}; }}

QLabel#preview {{ background: #000; color: {P['muted']}; border: 1px solid {P['border']}; border-radius: 8px; }}

QTabWidget::pane {{ border: none; background: {P['bg']}; padding-top: 8px; }}
QTabBar::tab {{ background: transparent; color: {P['muted']}; padding: 7px 16px; margin-right: 4px;
                border-radius: 8px; font-weight: 500; }}
QTabBar::tab:selected {{ background: {P['surface2']}; color: {P['text']}; }}
QTabBar::tab:hover:!selected {{ color: {P['text']}; }}

QPushButton {{ background: {P['surface2']}; color: {P['text']}; border: 1px solid {P['border']};
               border-radius: 8px; padding: 7px 14px; font-weight: 500; }}
QPushButton:hover {{ border-color: {P['muted']}; }}
QPushButton:pressed {{ background: {P['border']}; }}
QPushButton:disabled {{ color: {P['muted']}; border-color: {P['surface2']}; }}
QPushButton#primary {{ background: {P['accent']}; color: {P['bg']}; border-color: {P['accent']}; }}
QPushButton#primary:hover {{ background: #74d4c8; }}
QPushButton#primary:disabled {{ background: {P['accent_dim']}; color: {P['muted']}; border-color: {P['accent_dim']}; }}

QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid {P['muted']};
                        background: {P['surface']}; }}
QCheckBox::indicator:checked {{ background: {P['accent']}; border-color: {P['accent']}; }}
QCheckBox::indicator:disabled {{ border-color: {P['border']}; }}

QComboBox, QSpinBox {{ background: {P['surface2']}; border: 1px solid {P['border']}; border-radius: 6px;
                       padding: 4px 8px; min-height: 20px; }}
QComboBox:hover, QSpinBox:hover {{ border-color: {P['muted']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {P['surface2']}; color: {P['text']}; border: 1px solid {P['border']};
                               selection-background-color: {P['accent_dim']}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 14px; border: none; background: transparent; }}

QListWidget {{ background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 8px; padding: 4px;
               font-family: Menlo, monospace; font-size: 12px; outline: none; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:first {{ color: {P['text']}; }}
QListWidget::item:selected {{ background: {P['surface2']}; }}

QPlainTextEdit {{ background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 8px;
                  font-family: Menlo, monospace; font-size: 12px; padding: 6px; }}

QProgressBar {{ background: {P['surface2']}; border: none; border-radius: 3px; height: 6px; max-height: 6px;
                text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {P['accent']}; border-radius: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {P['border']}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(P["bg"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(P["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(P["surface"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(P["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(P["surface2"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(P["text"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(P["accent_dim"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(P["text"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(P["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property (role, ok, recording) changes."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class ActivityMeter(QWidget):
    """Seven rounded bars of lip activity, newest on the right; lit in the accent while recording."""

    def __init__(self, bars: int = 7, parent=None):
        super().__init__(parent)
        self.levels = [0.08] * bars
        self.recording = False
        self.setFixedHeight(26)
        self.setMinimumWidth(bars * 14)

    def set_levels(self, levels: list[float], recording: bool = False) -> None:
        self.levels, self.recording = list(levels), recording
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        n = len(self.levels)
        gap, w = 5.0, (self.width() - 5.0 * (n - 1)) / n
        h = self.height()
        for i, v in enumerate(self.levels):
            bar_h = max(4.0, v * h)
            x = i * (w + gap)
            p.setBrush(QColor(P["surface2"]))
            p.drawRoundedRect(QRectF(x, 0, w, h), 3, 3)
            p.setBrush(QColor(P["rec"] if self.recording else (P["accent"] if i == n - 1 else P["muted"])))
            p.drawRoundedRect(QRectF(x, h - bar_h, w, bar_h), 3, 3)
        p.end()
