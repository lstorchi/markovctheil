from PyQt4 import QtCore, QtGui
import sys

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils

class plohitwin (QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(plohitwin, self).__init__(parent)

        self.resize(640, 480) 
        self.setWindowTitle('Plots')

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        maindialog = QtGui.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)


