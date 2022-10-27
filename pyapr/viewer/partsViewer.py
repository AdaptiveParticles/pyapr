from pyqtgraph.Qt import QtCore, QtWidgets
from _pyaprwrapper.data_containers import APR, ByteParticles, ShortParticles, FloatParticles, LongParticles
from _pyaprwrapper.viewer import fill_slice, fill_slice_level, min_occupied_level
from ..utils import particles_to_type
from .._common import _check_input
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from typing import Union


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

        self.pg_win = pg.GraphicsView()
        self.view = pg.ViewBox()
        self.view.setAspectLocked()
        self.pg_win.setCentralItem(self.view)
        self.layout.addWidget(self.pg_win, 0, 0, 3, 1)

        # add a slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)

        self.slider.valueChanged.connect(self.valuechange)

        self.setGeometry(300, 300, self.full_size, self.full_size)

        self.layout.addWidget(self.slider, 1, 0)

        # add a histogram

        self.hist = pg.HistogramLUTWidget()

        self.layout.addWidget(self.hist, 0, 1)

        self.hist.item.sigLevelsChanged.connect(self.histogram_updated)

        # add a drop box for LUT selection

        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.move(20, 20)
        self.comboBox.addItem('bone')
        self.comboBox.addItem('viridis')
        self.comboBox.addItem('plasma')
        self.comboBox.addItem('inferno')
        self.comboBox.addItem('magma')
        self.comboBox.addItem('cividis')
        self.comboBox.addItem('Greys')
        self.comboBox.addItem('Greens')
        self.comboBox.addItem('Oranges')
        self.comboBox.addItem('Reds')
        self.comboBox.addItem('Pastel1')

        self.comboBox.currentTextChanged.connect(self.updatedLUT)

        # add a QLabel giving information on the current slice and the APR
        self.slice_info = QtWidgets.QLabel(self)

        self.slice_info.move(130, 20)
        self.slice_info.setFixedWidth(200)
        self.slice_info.setFixedHeight(45)

        # add a label for the current cursor position

        self.cursor = QtWidgets.QLabel(self)

        self.cursor.move(330, 20)
        self.cursor.setFixedWidth(260)
        self.cursor.setFixedHeight(45)

    def add_level_toggle(self):
        self.level_toggle = QtWidgets.QCheckBox(self)
        self.level_toggle.setText("View Level")
        self.level_toggle.move(625, 20)

        self.level_toggle.setChecked(False)

        self.level_toggle.stateChanged.connect(self.toggleLevel)

    def toggleLevel(self):
        force_update = self.current_view
        self.current_view = -pow(2, self.level_max-self.level_min+1)

        if self.level_toggle.isChecked():
            self.hist_on = False
            for l in range(self.level_min, self.level_max + 1):
                self.img_list[l].setLevels([self.level_min, self.level_max], True)
                self.zero_img.setLevels([0, 1], True)
        else:
            self.hist_on = True
            self.histogram_updated()

        self.update_slice(force_update)

    img_list = []

    current_view = 0

    array_int = np.array(1)
    aAPR_ref = 0
    parts_ref = 0

    dtype = 0

    x_num = 0
    z_num = 0
    y_num = 0

    array_list = []

    level_max = 0
    level_min = 0

    full_size = 900
    scale_sc = 10

    min_x = 0
    min_y = 0

    hist_min = 0
    hist_max = 1

    lut = 0
    lut_back = 0

    hist_on = True

    def updateSliceText(self, z):
        text_string = 'Slice: {}/{}, {}x{}\n' \
                      'level_min: {}, level_max: {}\n'.format(z+1, self.z_num, self.y_num, self.x_num,
                                                              self.level_min, self.level_max)

        self.slice_info.setText(text_string)

    def updatedLUT(self):
        # monitors the event of the drop box being manipulated
        self.setLUT(self.comboBox.currentText())

    def setLUT(self, string):

        call_dict = {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'cividis': plt.cm.cividis,
            'Greys': plt.cm.Greys,
            'Greens': plt.cm.Greens,
            'Oranges': plt.cm.Oranges,
            'Reds': plt.cm.Reds,
            'bone': plt.cm.bone,
            'Pastel1': plt.cm.Pastel1
        }

        # color map integration using LUT
        self.cmap = call_dict[string]

        self.lut = self.cmap(np.linspace(0.0, 1.0, 512))
        self.lut = self.lut * 255

        self.lut_back = self.lut.copy()
        self.lut[0, :] = 0
        self.lut_back[0, 3] = 255

        self.lut[1, :] = 0

        self.zero_img.setLookupTable(self.lut_back, True)

        for l in range(self.level_min, self.level_max + 1):
            self.img_list[l].setLookupTable(self.lut, True)

    def init_APR(self, aAPR, parts):
        self.aAPR_ref = aAPR
        self.parts_ref = parts

        self.dtype = particles_to_type(parts)

        self.z_num = aAPR.z_num(aAPR.level_max())
        self.x_num = aAPR.x_num(aAPR.level_max())
        self.y_num = aAPR.y_num(aAPR.level_max())
        self.level_max = aAPR.level_max()
        self.level_min = min_occupied_level(self.aAPR_ref)

        ## Set up the slide
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.z_num-1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider.setGeometry(int(0.05*self.full_size), int(0.97*self.full_size), int(0.95*self.full_size), 40)

        ## Viewer elements

        self.setWindowTitle('Demo APR Viewer')
        #

        self.view.setAspectLocked(True)

        self.view.setRange(QtCore.QRectF(0, 0, self.full_size, self.full_size))

        for i in range(0, self.level_max + 1):
            xl = aAPR.x_num(i)
            yl = aAPR.y_num(i)

            self.array_list.append(np.zeros([xl, yl], dtype=self.dtype))
            self.img_list.append(pg.ImageItem())

        #
        #   Init the images
        #

        max_x = 0
        max_y = 0

        for l in range(self.level_min, self.level_max + 1):
            sz = pow(2, self.level_max - l)
            img_sz_x = self.array_list[l].shape[1] * sz
            img_sz_y = self.array_list[l].shape[0] * sz
            max_x = max(max_x, img_sz_x)
            max_y = max(max_y, img_sz_y)

        #
        #   Setting the scale of the image to initialize
        #
        max_dim = max(max_x,max_y)
        self.scale_sc = self.full_size/max_dim

        max_x = max_x*self.scale_sc
        max_y = max_y*self.scale_sc

        self.zero_array = np.zeros([1, 1], dtype=self.dtype)
        self.zero_array[0, 0] = 0
        self.zero_img = pg.ImageItem(self.zero_array)
        self.view.addItem(self.zero_img)
        self.zero_img.setRect(QtCore.QRectF(self.min_x, self.min_y, max_x, max_y))

        for l in range(self.level_min, self.level_max + 1):
            self.view.addItem(self.img_list[l])

        self.setLUT('bone')

        self.current_view = -pow(2, self.level_max-self.level_min+1)
        self.update_slice(int(self.z_num*0.5))

        ## Setting up the histogram

        ## Needs to be updated to relay on a subsection of the particles
        arr = np.array(parts, copy=False)
        arr = arr.reshape((arr.shape[0], 1))

        ## then need to make it 2D, so it can be interpreted as an img;

        self.img_hist = pg.ImageItem(arr)
        self.hist.setImageItem(self.img_hist)

        ## Image hover event
        self.img_list[self.level_max].hoverEvent = self.imageHoverEvent

    def update_slice(self, new_view):

        if (new_view >= 0) & (new_view < self.z_num):
            # now update the view
            for l in range(self.level_min, self.level_max + 1):
                # loop over levels of the APR
                sz = pow(2, self.level_max - l)

                curr_z = int(new_view/sz)
                prev_z = int(self.current_view/sz)

                if prev_z != curr_z:

                    if self.level_toggle.isChecked():
                        fill_slice_level(self.aAPR_ref, self.parts_ref, self.array_list[l], curr_z, l)
                    else:
                        fill_slice(self.aAPR_ref, self.parts_ref, self.array_list[l], curr_z, l)

                    self.img_list[l].setImage(self.array_list[l], False)

                    img_sz_x = self.scale_sc * self.array_list[l].shape[1] * sz
                    img_sz_y = self.scale_sc * self.array_list[l].shape[0] * sz

                    self.img_list[l].setRect(QtCore.QRectF(self.min_x, self.min_y, img_sz_x, img_sz_y))

            self.current_view = new_view
            # make the slider reflect the new value
            self.slider.setValue(new_view)
            self.updateSliceText(new_view)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            # back a frame
            self.update_slice(self.current_view - 1)

        if event.key() == QtCore.Qt.Key_Right:
            # forward a frame
            self.update_slice(self.current_view + 1)

    def valuechange(self):
        size = self.slider.value()
        self.update_slice(size)

    def histogram_updated(self):

        if self.hist_on:
            hist_range = self.hist.item.getLevels()

            self.hist_min = hist_range[0]
            self.hist_max = hist_range[1]

            for l in range(self.level_min, self.level_max + 1):
                self.img_list[l].setLevels([self.hist_min,  self.hist_max], True)

                self.zero_img.setLevels([0,  1], True)

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """

        if event.isExit():
            return

        current_level = self.level_max

        data = self.array_list[self.level_max]

        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, data.shape[0] - 1))
        j = int(np.clip(j, 0, data.shape[1] - 1))
        val = data[i, j]

        k = self.current_view

        i_l = i
        j_l = j
        k_l = k

        while (val == 0) & (current_level > self.level_min):
            current_level -= 1
            i_l = int(i_l/2)
            j_l = int(j_l/2)
            k_l = int(k_l/2)
            val = self.array_list[current_level][i_l, j_l]

        text_string = 'x={}, y={}, z={}, value={}\n' \
                      'x_l={}, y_l={}, z_l={}, level={}\n'.format(j, i, k, val, j_l, i_l, k_l, current_level)

        self.cursor.setText(text_string)


def parts_viewer(apr: APR,
                 parts: Union[ByteParticles, ShortParticles, FloatParticles, LongParticles]):
    """
    Spawns an interactive 2D APR viewer.

    Parameters
    ----------
    apr: APR
        Input APR data structure.
    parts: ParticleData
        Input particle intensity values.
    """
    _check_input(apr, parts)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOption('imageAxisOrder', 'row-major')

    # Create window with GraphicsView widget
    win = MainWindow()
    win.add_level_toggle()
    win.init_APR(apr, parts)
    win.show()

    app.exec_()
