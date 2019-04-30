from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import basicutils

class plohitwin (QtWidgets.QMainWindow):

    def __init__(self, allratings, allratingsnins, parent=None):
        super(plohitwin, self).__init__(parent)

        self.resize(640, 480) 
        self.setWindowTitle('Empirical distributions')

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        maindialog = QtWidgets.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)
        
        k = 1
        for i in range(len(allratings)):
            ax = figure.add_subplot(2, 4, i+1)
            #ax.hold(False)
            ax.set_title("K="+str(k))
            ax.grid(True)
            ax.set_ylabel('f(bp)')
            ax.set_xlabel('bp')
            ax.hist(allratings[i], normed=False, \
                    bins=allratingsnins[i], \
                    facecolor='green')
            k = k + 1
        
        canvas.draw()

