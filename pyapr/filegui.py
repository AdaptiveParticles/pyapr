import pyqtgraph.Qt as Qt
import pyqtgraph as pg
import matplotlib.pyplot as plt
import numpy as np


class DoubleSlider(Qt.QtWidgets.QSlider):
    """
    Extends QSlider to allow floating-point values

    Adapted from Stack Overflow answer https://stackoverflow.com/a/50300848
    by user bfris (https://stackoverflow.com/users/9705687/bfris)
    """

    # create a signal that we can connect to if necessary
    doubleValueChanged = Qt.QtCore.pyqtSignal(float)

    def __init__(self, decimals=2, *args, **kwargs):
        super(DoubleSlider, self).__init__(*args, **kwargs)
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value()) / self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi


class CustomSlider:
    def __init__(self, window, label_name, decimals=0):

        if decimals < 0 or not isinstance(decimals, int):
            raise ValueError('CustomSlider initialized with \'decimals\'={}. Only non-negative integers are allowed.'.format(decimals))

        self.decimals = decimals

        self.slider = DoubleSlider(decimals, Qt.QtCore.Qt.Horizontal, window)
        self.maxBox = Qt.QtWidgets.QDoubleSpinBox(window, decimals=self.decimals)

        self.label = Qt.QtWidgets.QLabel(window)

        self.maxBox.setMaximum(64000)
        self.maxBox.setValue(300)

        self.win_ref = window
        self.label_name = label_name

        self.maxBox.valueChanged.connect(self.updateRange)
        self.slider.valueChanged.connect(self.updateText)

        self.slider.setValue(1)
        self.slider.setMaximum(self.maxBox.value())

        self.sz_label = 200
        self.sz_slider = 200
        self.sz_box = 90

    def move(self, loc1, loc2):

        self.label.move(loc1, loc2-5)
        self.label.setFixedWidth(self.sz_label)

        self.slider.move(loc1 + self.sz_label, loc2)
        self.slider.setFixedWidth(self.sz_slider)
        self.maxBox.move(loc1 + self.sz_slider + self.sz_label + 5, loc2-5)
        self.maxBox.setFixedWidth(self.sz_box)

    def updateRange(self):
        max_val = self.maxBox.value()
        self.slider.setMaximum(max_val)

    def connectSlider(self, function):
        self.slider.valueChanged.connect(function)

    def updateText(self):
        val_str = '{:.{prec}f}'.format(self.slider.value(), prec=self.decimals)
        text_str = self.label_name + ': ' + val_str
        self.label.setText(text_str)


class MainWindowImage(Qt.QtGui.QWidget):
    def __init__(self, slider_decimals=0):
        super(MainWindowImage, self).__init__()

        self.setMouseTracking(True)

        self.layout = Qt.QtGui.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setSpacing(0)

        self.pg_win = pg.GraphicsView()
        self.view = pg.ViewBox()
        self.view.setAspectLocked()
        self.pg_win.setCentralItem(self.view)
        self.layout.addWidget(self.pg_win, 0, 0, 3, 1)

        # add a slider
        self.slider = Qt.QtWidgets.QSlider(Qt.QtCore.Qt.Horizontal, self)

        self.slider.valueChanged.connect(self.valuechange)

        self.setGeometry(300, 300, self.full_size, self.full_size)

        self.layout.addWidget(self.slider, 1, 0)

        # add a histogram

        self.hist = pg.HistogramLUTWidget()

        self.layout.addWidget(self.hist, 0, 1)

        self.hist.item.sigLevelsChanged.connect(self.histogram_updated)

        # add a QLabel giving information on the current slice and the APR
        self.slice_info = Qt.QtGui.QLabel(self)

        self.slice_info.move(20, 20)
        self.slice_info.setFixedWidth(250)

        # add a label for the current cursor position

        self.cursor = Qt.QtGui.QLabel(self)

        self.cursor.move(20, 40)
        self.cursor.setFixedWidth(250)

        # add parameter tuning

        # create push button
        self.exit_button = Qt.QtWidgets.QPushButton('Use Parameters', self)
        self.exit_button.setFixedWidth(300)
        self.exit_button.move(300, 10)
        self.exit_button.clicked.connect(self.exitPressed)

        self.max_label = Qt.QtWidgets.QLabel(self)
        self.max_label.setText("Slider Max")
        self.max_label.move(610, 50)

        self.slider_grad = CustomSlider(self, "gradient threshold", decimals=slider_decimals)
        self.slider_grad.move(200, 80)
        self.slider_grad.connectSlider(self.valuechangeGrad)

        self.slider_sigma = CustomSlider(self, "sigma threshold", decimals=slider_decimals)
        self.slider_sigma.move(200, 110)
        self.slider_sigma.connectSlider(self.valuechangeSigma)

        self.slider_Ith = CustomSlider(self, "intensity threshold", decimals=slider_decimals)
        self.slider_Ith.move(200, 140)
        self.slider_Ith.connectSlider(self.valuechangeIth)

        # add a label for the current cursor position

    current_view = 0

    array_int = np.array(1)
    img_ref = 0
    par_ref = 0

    x_num = 0
    z_num = 0
    y_num = 0

    full_size = 1000
    scale_sc = 10

    min_x = 0
    min_y = 0

    hist_min = 0
    hist_max = 200

    lut = 0
    lut_back = 0

    #parameters to be played with
    grad_th = 0
    app_ref = 0

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """

        if event.isExit():
            return

        data = self.img_ref[self.current_view, :, :]

        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, data.shape[0] - 1))
        j = int(np.clip(j, 0, data.shape[1] - 1))
        val = data[i, j]

        text_string = 'x={}, y={}, z={}, value={}\n'.format(j, i, self.current_view, val)
        self.cursor.setText(text_string)

    def exitPressed(self):
        self.app_ref.exit()

    def updateSliceText(self, z):
        text_string = 'Slice: {}/{}, {}x{}\n'.format(z+1, self.z_num, self.y_num, self.x_num)
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

        self.img_I.setLookupTable(self.lut_back, True)

        self.red = call_dict['Reds']
        self.lut_mask = self.red(np.linspace(0.0, 1.0, 512))*255
        self.lut_mask[0, :] = 0
        self.lut_mask[1, :] = 255

        self.img_I_ds.setLookupTable(self.lut_mask, True)

        self.img_I_ds.setImage(None, levels=(self.apr_ref.level_max()-2, self.apr_ref.level_max()), opacity=0.5)

    def update_slice(self, new_view):

        if (new_view >= 0) & (new_view < self.z_num):
            # now update the view

            self.img_I.setImage(self.img_ref[new_view, :, :], False)

            self.converter.get_level_slice(int(new_view/2), self.img_ds, self.par_ref, self.apr_ref)

            self.img_I_ds.setImage(self.img_ds, False)

            self.current_view = new_view
            # make the slider reflect the new value
            self.slider.setValue(new_view)
            self.updateSliceText(new_view)

    def keyPressEvent(self, event):
        if event.key() == Qt.QtCore.Qt.Key_Left:
            # back a frame
            self.update_slice(self.current_view - 1)

        if event.key() == Qt.QtCore.Qt.Key_Right:
            # forward a frame
            self.update_slice(self.current_view + 1)

    def valuechange(self):
        size = self.slider.value()
        self.update_slice(size)

    def valuechangeGrad(self):
        size = self.slider_grad.slider.value()
        self.par_ref.grad_th = size
        self.update_slice(self.current_view)

    def valuechangeSigma(self):
        size = self.slider_sigma.slider.value()
        self.par_ref.sigma_th = size
        self.update_slice(self.current_view)

    def valuechangeIth(self):
        size = self.slider_Ith.slider.value()
        self.par_ref.Ip_th = size
        self.update_slice(self.current_view)

    def histogram_updated(self):

        hist_range = self.hist.item.getLevels()

        self.hist_min = hist_range[0]
        self.hist_max = hist_range[1]

        #self.img_I.setLevels([self.hist_min,  self.hist_max], True)

    def set_image(self, img, converter):

        self.img_I = pg.ImageItem(img[0, :, :])
        self.view.addItem(self.img_I)

        self.img_ref = img

        self.converter = converter

        self.z_num = img.shape[0]

        self.y_num = img.shape[1]
        self.x_num = img.shape[2]

        self.z_num_ds = int((img.shape[0]+1) / 2)
        self.y_num_ds = int((img.shape[1]+1) / 2)
        self.x_num_ds = int((img.shape[2]+1) / 2)

        self.img_ds = np.zeros((self.y_num_ds, self.x_num_ds), dtype=np.float32)

        self.par_ref = converter.get_parameters()

        self.par_ref.grad_th = self.grad_th

        converter.get_level_slice(0, self.img_ds, self.par_ref, self.apr_ref)

        self.img_I_ds = pg.ImageItem(self.img_ds)
        self.view.addItem(self.img_I_ds)

        self.hist.setImageItem(self.img_I)

        self.img_I_ds.setRect(Qt.QtCore.QRectF(self.min_x, self.min_y, self.x_num_ds*2, self.y_num_ds*2))
        self.img_I.setRect(Qt.QtCore.QRectF(self.min_x, self.min_y, self.x_num, self.y_num))

        ## Set up the z slider
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.z_num - 1)
        self.slider.setTickPosition(Qt.QtWidgets.QSlider.TicksBothSides)
        self.slider.setGeometry(0.05 * self.full_size, 0.97 * self.full_size, 0.95 * self.full_size, 40)

        self.setLUT('viridis')

        ## Image hover event
        self.img_I.hoverEvent = self.imageHoverEvent
        self.update_slice(int(self.z_num/2))

    def closeEvent(self, event):
        self.pg_win.close()


class InteractiveIO:
    def __init__(self):
        # class methods require a QApplication instance - this helps to avoid multiple instances...
        self.app = Qt.QtGui.QApplication.instance()
        if self.app is None:
            self.app = Qt.QtGui.QApplication([])

    @staticmethod
    def get_tiff_file_name():
        print("Please select an input image file (TIFF)")
        file_name = Qt.QtGui.QFileDialog.getOpenFileName(None, "Open Tiff", "~", "(*.tif *.tiff)")
        return file_name[0]

    @staticmethod
    def get_apr_file_name():
        print("Please select an input APR file (HDF5)")
        file_name = Qt.QtGui.QFileDialog.getOpenFileName(None, "Open APR", "", "(*.apr *.h5)")
        return file_name[0]

    @staticmethod
    def save_apr_file_name(default_name='output.apr'):
        file_name = Qt.QtGui.QFileDialog.getSaveFileName(None, "Save APR", default_name, "(*.apr *.h5)")
        return file_name[0]

    @staticmethod
    def save_tiff_file_name(default_name='output.tif'):
        file_name = Qt.QtGui.QFileDialog.getSaveFileName(None, "Save TIFF", default_name, "(*.tif *.tiff)")
        return file_name[0]

    def interactive_apr(self, converter, apr, img, slider_decimals=2):

        converter.get_apr_step1(apr, img)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')

        # Create window with GraphicsView widget
        win = MainWindowImage(slider_decimals=slider_decimals)
        win.show()
        win.apr_ref = apr
        win.app_ref = self.app
        win.set_image(img, converter)

        self.app.exec_()
        win.close()

        # now compute the APR

        print("---------------------------------")
        print("Using the following parameters:")
        print("grad_th = {}, sigma_th = {}, Ip_th = {}".format(win.par_ref.grad_th,
                                                               win.par_ref.sigma_th, win.par_ref.Ip_th))
        print("---------------------------------")

        converter.get_apr_step2(apr, win.par_ref)
        return None

    def find_parameters_interactive(self, converter, apr, img, slider_decimals=2):

        converter.get_apr_step1(apr, img)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')

        # Create window with GraphicsView widget
        win = MainWindowImage(slider_decimals=slider_decimals)
        win.show()
        win.apr_ref = apr
        win.app_ref = self.app
        win.set_image(img, converter)

        # Run the app
        self.app.exec_()
        win.close()

        # Return the Parameters
        return win.par_ref
