import os
import pytest
from pytestqt import qtbot
from pyqtgraph.Qt import QtCore
import pyapr
from .helpers import load_test_apr


@pytest.mark.skipif('DISPLAY' not in os.environ, reason='requires display')
def test_viewer(qtbot):
    apr, parts = load_test_apr(3)

    # launch viewer
    win = pyapr.viewer.MainWindow()
    win.add_level_toggle()
    win.init_APR(apr, parts)
    qtbot.add_widget(win)

    # change LUT and view mode
    qtbot.keyClicks(win.comboBox, "magma")
    qtbot.mouseClick(win.level_toggle, QtCore.Qt.LeftButton)

    # change z-slice
    z_prev = win.slider.value()
    qtbot.mouseClick(win.slider, QtCore.Qt.LeftButton, pos=win.slider.rect().center() + QtCore.QPoint(10, 0))
    assert win.slider.value() != z_prev

    # try to trigger mouse hover event (does not seem to work)
    qtbot.mouseMove(win.pg_win, win.pg_win.rect().center(), delay=100)
