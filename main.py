"""
main.py
========
App entry point.

    python main.py

Loads the bundled Nunito font, applies the QSS theme, and shows MainWindow.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFontDatabase, QFont

from ui.theme import build_stylesheet
from ui.main_window import MainWindow

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")


def _load_bundled_fonts() -> str:
    """
    Loads the Nunito variable fonts from assets/fonts and returns the family
    name Qt registered them under. Falls back to "Nunito" (or Qt's default
    if that's not installed anywhere) if loading fails for any reason —
    this must never crash the app on startup.
    """
    family = "Nunito"
    for filename in ("Nunito-Variable.ttf", "Nunito-Italic-Variable.ttf"):
        path = os.path.join(FONT_DIR, filename)
        if os.path.exists(path):
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    family = families[0]
    return family


def main():
    app = QApplication(sys.argv)

    family = _load_bundled_fonts()
    app.setFont(QFont(family, 10))
    app.setStyleSheet(build_stylesheet(font_family=family))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
