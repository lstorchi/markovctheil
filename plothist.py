from PyQt4 import QtCore, QtGui
import sys

sys.path.append("./module")
import basicutils

class plohitwin (QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(plohitwin, self).__init__(parent)

        self.resize(640, 480) 
        self.setWindowTitle('Plots')

    def closedialog(self):
        self.close()
