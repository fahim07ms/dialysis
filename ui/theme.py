"""
theme.py
========
All colors, spacing and the QSS (Qt's CSS-like stylesheet language) live
here in ONE place. If you ever want to re-theme the whole app, this is the
only file you should need to touch.

No signal-processing logic belongs in this file.
"""

# ---------------------------------------------------------------------------
# Palette — sampled from the reference screenshot (Pixel-style quick settings)
# ---------------------------------------------------------------------------

BG = "#1B1C1F"            # app background (near-black)
SURFACE = "#242529"       # cards / panels, one step lighter than BG
SURFACE_HOVER = "#2C2D32"
SURFACE_ALT = "#3A3B40"   # e.g. the greyed-out chevron chip in the screenshot

PRIMARY = "#54E0D0"       # the teal accent (sliders, active chip, focus ring)
PRIMARY_DIM = "#3AA89C"   # pressed / darker teal
ON_PRIMARY = "#08201D"    # dark text drawn on top of the teal fill

TEXT = "#F2F3F5"          # primary text, near-white
TEXT_MUTED = "#9AA0A6"    # secondary text (subtitles, hints)
TEXT_DISABLED = "#5A5C61"

BORDER = "#33343A"
TRACK = "#4A4B50"         # slider track (unfilled portion)

RADIUS_LG = 20             # big pill / card radius
RADIUS_MD = 14
RADIUS_SM = 10


def build_stylesheet(font_family: str = "Nunito") -> str:
    """
    Returns the full QSS string for the application.

    font_family: the family name QFontDatabase gave us back after loading
                 the bundled Nunito .ttf (see main.py). Falling back to the
                 literal string "Nunito" is safe — Qt silently falls back
                 to a default font if the family isn't found, it won't crash.
    """
    return f"""
    /* ---------- Global ---------- */
    * {{
        font-family: "{font_family}";
        outline: none;
    }}

    QMainWindow, QWidget#RootBackground {{
        background-color: {BG};
    }}

    QLabel {{
        color: {TEXT};
        background: transparent;
    }}

    QLabel[role="muted"] {{
        color: {TEXT_MUTED};
        font-size: 13px;
    }}

    QLabel[role="title"] {{
        color: {TEXT};
        font-size: 20px;
        font-weight: 800;
    }}

    QLabel[role="section"] {{
        color: {TEXT_MUTED};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    /* ---------- Cards / panels ---------- */
    QFrame[role="card"] {{
        background-color: {SURFACE};
        border-radius: {RADIUS_LG}px;
        border: 1px solid {BORDER};
    }}

    /* ---------- Keypad buttons ---------- */
    QPushButton[role="key"] {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_MD}px;
        font-size: 22px;
        font-weight: 700;
        min-height: 56px;
        min-width: 56px;
    }}
    QPushButton[role="key"]:hover {{
        background-color: {SURFACE_HOVER};
        border: 1px solid {PRIMARY_DIM};
    }}
    QPushButton[role="key"]:pressed {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border: 1px solid {PRIMARY};
    }}

    /* ---------- Pill buttons (Play / Export / toggles) ---------- */
    QPushButton[role="pill-primary"] {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border: none;
        border-radius: {RADIUS_LG}px;
        font-size: 14px;
        font-weight: 800;
        padding: 12px 20px;
    }}
    QPushButton[role="pill-primary"]:hover {{
        background-color: #6FE8DA;
    }}
    QPushButton[role="pill-primary"]:pressed {{
        background-color: {PRIMARY_DIM};
    }}
    QPushButton[role="pill-primary"]:disabled {{
        background-color: {SURFACE_ALT};
        color: {TEXT_DISABLED};
    }}

    QPushButton[role="pill-outline"] {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_LG}px;
        font-size: 14px;
        font-weight: 700;
        padding: 12px 20px;
    }}
    QPushButton[role="pill-outline"]:hover {{
        background-color: {SURFACE_HOVER};
    }}
    QPushButton[role="pill-outline"]:pressed {{
        background-color: {SURFACE_ALT};
    }}

    /* ---------- Text field ---------- */
    QLineEdit {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_MD}px;
        padding: 10px 14px;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 3px;
    }}
    QLineEdit:focus {{
        border: 1px solid {PRIMARY};
    }}

    /* ---------- Sliders ---------- */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {TRACK};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {PRIMARY};
        border-radius: 2px;
    }}
    QSlider::add-page:horizontal {{
        background: {TRACK};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {TEXT};
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {PRIMARY};
    }}

    /* ---------- Tabs (right-hand analysis panel) ---------- */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        border-radius: {RADIUS_LG}px;
        background-color: {SURFACE};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT_MUTED};
        font-weight: 700;
        font-size: 13px;
        padding: 10px 16px;
        margin-right: 4px;
        border-top-left-radius: {RADIUS_SM}px;
        border-top-right-radius: {RADIUS_SM}px;
    }}
    QTabBar::tab:selected {{
        color: {ON_PRIMARY};
        background: {PRIMARY};
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        background: {SURFACE_HOVER};
    }}

    /* ---------- Combo box (algorithm selector) ---------- */
    QComboBox {{
        background-color: {SURFACE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_MD}px;
        padding: 8px 12px;
        font-weight: 700;
    }}
    QComboBox:hover {{
        border: 1px solid {PRIMARY_DIM};
    }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE};
        color: {TEXT};
        selection-background-color: {PRIMARY};
        selection-color: {ON_PRIMARY};
        border: 1px solid {BORDER};
        outline: none;
    }}

    /* ---------- Scrollbars (keep them slim & unobtrusive) ---------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {SURFACE_ALT};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    /* ---------- Splitter ---------- */
    QSplitter::handle {{
        background: {BG};
        width: 12px;
    }}
    """
