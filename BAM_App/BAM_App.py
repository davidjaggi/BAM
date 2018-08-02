# -*- coding: utf-8 -*-

"""
Bancroft Application

Author: David Jaggi
Website: www.davidjaggi.com
Last edited: August 2018
"""
import sys
import os

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtGui import QIcon
from PyQt5 import uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib import rcParams


class BAM_App(QMainWindow):
    def __init__(self, parent = None):
        QMainWindow.__init__(self,parent)
        uic.loadUi('C:/Users/David Jaggi/Documents/GitHub/BAM/BAM_App/userInterface.ui', self)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BAM_App()
    window.show()
    app.exec_()