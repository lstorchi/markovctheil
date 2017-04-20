import sys
from PyQt4 import QtGui, QtCore 

class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        inputfile_ = ""

        QtGui.QMainWindow.__init__(self) 
        self.resize(350, 250) 
        self.setWindowTitle('Markov QT')
        self.statusBar().showMessage('Markov started') 

        ofile = QtGui.QAction(QtGui.QIcon("icons/open.png"), "Open", self)
        ofile.setShortcut("Ctrl+O")
        ofile.setStatusTip("Open file")
        ofile.connect(ofile, QtCore.SIGNAL('triggered()'), self.openfile)

        sep = QtGui.QAction(self)
        sep.setSeparator(True)

        quit = QtGui.QAction(QtGui.QIcon("icons/cancel.png"), "Quit", self)
        quit.setShortcut("Ctrl+Q")
        quit.setStatusTip("Quit application")
        self.connect(quit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))

        self.statusBar().show()

        menuB = self.menuBar()
        
        file = menuB.addMenu('&File')
        file.addAction(ofile)
        file.addAction(sep)
        file.addAction(quit)

    def openfile(self):
        inputfile_ = QtGui.QFileDialog.getOpenFileName(self) 


app = QtGui.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())
