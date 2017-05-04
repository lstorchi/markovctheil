from PyQt4 import QtCore, QtGui
import sys
import os

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import basicutils

class matable (QtGui.QMainWindow):

    def __init__(self, mat, name, parent=None):
        super(matable, self).__init__(parent)

        self.__mat__ = []

        self.resize(640, 480) 
        self.setWindowTitle(name)

        savefile = QtGui.QAction(QtGui.QIcon("icons/save.png"), "Save", self)
        savefile.setStatusTip("Save file")
        savefile.connect(savefile , QtCore.SIGNAL('triggered()'), \
         self.savefile)
        savefile.setEnabled(True)

        quit = QtGui.QAction(QtGui.QIcon("icons/cancel.png"), "Quit", self)
        quit.setShortcut("Ctrl+Q")
        quit.setStatusTip("Quit application")
        self.connect(quit, QtCore.SIGNAL('triggered()'), self.closedialog)

        menubar = self.menuBar()
        
        file = menubar.addMenu('&File')

        file.addAction(savefile)
        file.addAction(quit)

        self.__table__ = QtGui.QTableWidget(len(mat), len(mat[0]), self) 

        for i in range(mat.shape[0]):
            self.__mat__.append([])
            for j in range(mat.shape[1]):
                newitem = QtGui.QTableWidgetItem()
                newitem.setText(str(mat[i, j]))
                # not editable
                newitem.setFlags(newitem.flags() ^ QtCore.Qt.ItemIsEditable);
                self.__mat__[i].append(mat[i, j])
                self.__table__.setItem(i, j, newitem)

        self.__table__.resizeColumnsToContents()
        self.__table__.resizeRowsToContents()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.__table__ )

        maindialog = QtGui.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)

    def closedialog(self):

        # todo editable matrix
        for i in range(len(self.__mat__ )):
            for j in range(len(self.__mat__[0])):
                dat = str(self.__table__.itemAt(i, j).text())
                self.__mat__[i][j] = float(dat)

        self.close()
        return 

    def get_data(self):

        # todo editable matrix
        return self.__mat__

    def savefile(self):

        tosave = QtGui.QFileDialog.getSaveFileName(self) 
        if os.path.exists(str(tosave)):
            os.remove(str(tosave))
            
        outf = open(str(tosave), "w")

        for i in range(len(self.__mat__)):
            for j in range(len(self.__mat__[0])):
                outf.write("%f "%(self.__mat__[i][j]))
            outf.write("\n")

        outf.close()
