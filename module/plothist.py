from PyQt4 import QtCore, QtGui
import sys

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import basicutils

class plohitwin (QtGui.QMainWindow):

    def __init__(self, allratings, allratingsnins, parent=None):
        super(plohitwin, self).__init__(parent)

        self.resize(640, 480) 
        self.setWindowTitle('CS distributions')

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        maindialog = QtGui.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)
        
        k = 1
        for i in range(len(allratings)):
            ax = figure.add_subplot(2, 4, i+1)
            #ax.hold(False)
            ax.set_title("K="+str(k))
            ax.grid(True)
            ax.set_ylabel('f(BP)')
            ax.set_xlabel('BP(%)')
            ax.hist(allratings[i], normed=False, \
                    bins=allratingsnins[i], \
                    facecolor='green')
            k = k + 1
        
        canvas.draw()

