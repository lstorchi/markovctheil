import sys
from PyQt4 import QtGui, QtCore 

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self) 
        self.resize(350, 250) 
        self.setWindowTitle('Markov QT')
        self.statusBar().showMessage('Markov started') 

app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
