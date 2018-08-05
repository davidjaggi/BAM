# -*- coding: utf-8 -*-

"""
Bancroft Application

Author: David Jaggi
Website: www.davidjaggi.com
Last edited: August 2018
"""
import sys
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import uic

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure

from MplWidget import MatplotlibWidget

import numpy as np


class BAM_App(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self,parent)
        uic.loadUi('C:/Users/David Jaggi/Documents/GitHub/BAM/BAM_App/userInterface.ui', self)
    
        self.plotData.clicked.connect(self.plot_data)
        self.clearPlot.clicked.connect(self.clear_plot)

        self.actionOpen.triggered.connect(self.openFileNameDialog)
        self.actionQuit.triggered.connect(self.fileQuit)
        self.actionInfo.triggered.connect(self.about)
    
    def plot_data(self):
        x=np.random.rand(100)
        y=np.random.rand(100)
        self.plotWidget.axes.plot(x,y)
        self.plotWidget.draw()
    
    def clear_plot(self):
        self.plotWidget.axes.clear()
        self.plotWidget.draw()

    def openFileNameDialog(self):
        try: 
            filename = QtWidgets.QFileDialog.getOpenFileName(
               self, 'Open File', '', 'Images (*.png *.xpm *.jpg)',
               None, QtGui.QFileDialog.DontUseNativeDialog)
            self.lineEdit.setText(filename)
        except:
            pass
    

    
    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = BAM_App()
    window.show()
    app.exec_()
