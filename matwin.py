from PyQt4 import QtCore, QtGui
import sys

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

sys.path.append("./module")
import basicutils

class matable (QtGui.QMainWindow):

    def __init__(self, mat, name, parent=None):
        super(matable, self).__init__(parent)

        self.resize(640, 480) 
        self.setWindowTitle(name)

        self.__mat__ = []

        self.__table__ = QtGui.QTableWidget(len(mat), len(mat[0]), self) 

        for i in range(len(mat)):
            self.__mat__.append([])
            for j in range(len(mat[0])):
                newitem = QtGui.QTableWidgetItem()
                newitem.setText(str(mat[i, j]))
                # not editable
                newitem.setFlags(newitem.flags() ^ QtCore.Qt.ItemIsEditable);
                self.__mat__[i].append(mat[i, j])
                self.__table__.setItem(i, j, newitem)

        self.__table__.resizeColumnsToContents()
        self.__table__.resizeRowsToContents()

        okbutton = QtGui.QPushButton('Ok');
        okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.__table__ )
        layout.addWidget(okbutton)

        maindialog = QtGui.QWidget()
        maindialog.setLayout(layout)

        self.setCentralWidget(maindialog)

    def closedialog(self):

        # todo editable matrix
        for i in range(len(self.__mat__)):
            for j in range(len(self.__mat__[0])):
                dat = str(self.__table__.itemAt(i, j).text())
                self.__mat__[i][j] = float(dat)

        self.close()
        return 

    def get_data(self):

        # todo editable matrix
        return self.__mat__
