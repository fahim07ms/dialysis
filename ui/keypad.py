"""
keypad.py
=========
The 4x4 virtual DTMF keypad. Pure UI — it only knows about button layout
and emits a Qt signal when a key is pressed. It has no idea what a sine
wave is; that's core/dsp_interface.py's job.
"""

from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt

# Same layout as core.dsp_interface.KEYPAD_LAYOUT, kept separate on purpose:
# the UI shouldn't need to import DSP code just to draw buttons.
KEYPAD_ROWS = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]


class Keypad(QWidget):
    """Emits `digit_pressed(str)` whenever a key is clicked."""

    digit_pressed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setSpacing(10)

        for row_i, row in enumerate(KEYPAD_ROWS):
            for col_i, digit in enumerate(row):
                btn = QPushButton(digit)
                btn.setProperty("role", "key")
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked, d=digit: self.digit_pressed.emit(d))
                grid.addWidget(btn, row_i, col_i)

        self.setLayout(grid)
