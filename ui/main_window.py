"""
main_window.py
================
Top-level window. Just assembles LeftPanel + RightPanel into a splitter.
No math, no device I/O — pure layout.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter
from PyQt6.QtCore import Qt

from ui.left_panel import LeftPanel
from ui.right_panel import RightPanel


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dialysis — DTMF Signal Analysis Studio")
        self.resize(1180, 720)
        self.setMinimumSize(900, 600)

        root = QWidget()
        root.setObjectName("RootBackground")
        self.setCentralWidget(root)

        outer_layout = QHBoxLayout(root)
        outer_layout.setContentsMargins(20, 20, 20, 20)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.left_panel = LeftPanel()
        self.right_panel = RightPanel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 800])

        outer_layout.addWidget(splitter)

        # Whenever the left panel produces new audio (play or import), push
        # it straight into the Waveform tab so it updates live.
        self.left_panel.audio_ready.connect(self.right_panel.waveform_tab.update_audio)

        # While audio is actually playing, stream partial samples so the
        # waveform/spectrum appear to grow in sync with playback.
        self.left_panel.playback_progress.connect(self.right_panel.waveform_tab.update_audio_live)
