from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from _pyaprwrapper.data_containers import APR, ShortParticles
from _pyaprwrapper.viewer import compress_and_fill_slice
from .partsViewer import MainWindow
from .._common import _check_input


class CustomSlider:
    def __init__(self, window, label_name):

        self.win_ref = window
        self.label_name = label_name

        self.label = QtWidgets.QLabel(window)

        self.maxBox = QtWidgets.QSpinBox(window)
        self.maxBox.setMaximum(64000)
        self.maxBox.setValue(100)
        self.maxBox.valueChanged.connect(self.updateRange)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, window)
        self.slider.valueChanged.connect(self.updateText)
        self.slider.setValue(1)

        self.sz_label = 120
        self.sz_slider = 200
        self.sz_box = 75

    def move(self, loc1, loc2):

        self.label.move(loc1, loc2-5)
        self.label.setFixedWidth(self.sz_label)

        self.slider.move(loc1 + self.sz_label, loc2)
        self.slider.setFixedWidth(self.sz_slider)
        self.maxBox.move(loc1 + self.sz_slider + self.sz_label + 5, loc2-5)
        self.maxBox.setFixedWidth(self.sz_box)

    def updateRange(self):
        max = self.maxBox.value()
        self.slider.setMaximum(max)
        self.slider.setTickInterval(1)

    def connectSlider(self, function):
        self.slider.valueChanged.connect(function)

    def updateText(self):
        text_str = self.label_name + ": " + str(self.slider.value())
        self.label.setText(text_str)


class CompressWindow(MainWindow):

    def __init__(self):
        super(CompressWindow, self).__init__()

        self.exit_button = QtWidgets.QPushButton('Use Parameters', self)
        self.exit_button.setFixedWidth(300)
        self.exit_button.move(500, 10)
        self.exit_button.clicked.connect(self.exitPressed)

        self.max_label = QtWidgets.QLabel(self)
        self.max_label.setText("Slider Max")
        self.max_label.move(520, 40)

        self.slider_q = CustomSlider(self, "quantization")
        self.slider_q.move(200, 70)
        self.slider_q.connectSlider(self.valuechangeQ)
        self.slider_q.maxBox.setValue(20)
        self.slider_q.slider.setSingleStep(0.1)

        self.slider_B = CustomSlider(self, "background")
        self.slider_B.move(200, 100)
        self.slider_B.connectSlider(self.valuechangeB)
        self.slider_B.maxBox.setValue(1000)

        self.toggle_on = QtWidgets.QCheckBox(self)
        self.toggle_on.setText("Compress")
        self.toggle_on.move(605, 65)
        self.toggle_on.setChecked(True)
        self.toggle_on.stateChanged.connect(self.toggleCompression)

    def toggleCompression(self):
        if self.toggle_on.isChecked():
            self.valuechangeQ()
            self.valuechangeB()
        else:
            self.parts_ref.set_quantization_factor(0)
            force_update = self.current_view
            self.current_view = -1
            self.update_slice(force_update)

    def exitPressed(self):
        self.app_ref.exit()

    def valuechangeQ(self):
        if self.toggle_on.isChecked():
            size = self.slider_q.slider.value()
            self.parts_ref.set_quantization_factor(size)
            force_update = self.current_view
            self.current_view = -1
            self.update_slice(force_update)

    def valuechangeB(self):
        if self.toggle_on.isChecked():
            size = self.slider_B.slider.value()
            self.parts_ref.set_background(size)
            force_update = self.current_view
            self.current_view = -1
            self.update_slice(force_update)

    def update_slice(self, new_view):
        if (new_view >= 0) & (new_view < self.z_num):
            # now update the view
            for l in range(self.level_min, self.level_max + 1):
                # loop over levels of the APR
                sz = pow(2, self.level_max - l)

                curr_z = int(new_view/sz)
                prev_z = int(self.current_view/sz)

                if prev_z != curr_z:
                    compress_and_fill_slice(self.aAPR_ref, self.parts_ref, self.array_list[l], curr_z, l)

                    self.img_list[l].setImage(self.array_list[l], False)

                    img_sz_x = self.scale_sc * self.array_list[l].shape[1] * sz
                    img_sz_y = self.scale_sc * self.array_list[l].shape[0] * sz

                    self.img_list[l].setRect(QtCore.QRectF(self.min_x, self.min_y, img_sz_x, img_sz_y))

            self.current_view = new_view
            # make the slider reflect the new value
            self.slider.setValue(new_view)
            self.updateSliceText(new_view)


def interactive_compression(apr: APR,
                            parts: ShortParticles):
    """
    Spawns a viewer to interactively find compression parameters for particle intensities.
    Slide the `quantization` and `background` sliders to change the compression behavior.

    Note: the parameters are saved in the input `parts` object, and used in subsequent I/O calls, e.g. `pyapr.io.write`.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ShortParticles
        Input particle intensity values.
    """
    _check_input(apr, parts, (ShortParticles,))
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOption('imageAxisOrder', 'row-major')

    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])

    ## Create window with GraphicsView widget
    win = CompressWindow()
    win.app_ref = app
    win.init_APR(apr, parts)

    win.show()
    app.exec_()

    #turn on
    parts.set_compression_type(1)
    return None

