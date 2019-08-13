
import PyQt5.QtCore as qtcore
import PyQt5.QtGui as qtgui

import pyapr

class InteractiveIO():

    def __init__(self):
        super(InteractiveIO, self).__init__()


    def get_tiff_file_name(self):

        app = qtgui.QApplication([])

        file_name = qtgui.QFileDialog.getOpenFileName(None, "Open Tiff", "~", "(*.tif *.tiff)")

        return file_name[0]


    def get_apr_file_name(self):

        app = qtgui.QApplication([])

        file_name = qtgui.QFileDialog.getOpenFileName(None, "Open APR", "~", "(*.apr *.h5)")

        return file_name[0]


    def save_apr_file_name(self):

        app = qtgui.QApplication([])

        file_name = qtgui.QFileDialog.getSaveFileName(None, "Save APR", "~", "(*.apr *.h5)")

        return file_name[0]


    def interactive_apr(self, converter, apr, img):




        return None