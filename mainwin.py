from PyQt4 import QtGui, QtCore 
import options

class main_window(QtGui.QMainWindow):
    
    def __init__(self):
        self.__inputfile__ = ""
        
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

        run = QtGui.QAction(QtGui.QIcon("icons/run.png"), "Run", self)
        run.setShortcut("Ctrl+R")
        run.setStatusTip("Run")
        self.connect(run, QtCore.SIGNAL('triggered()'), self.mainrun)

        self.statusBar().show()

        menubar = self.menuBar()
        
        file = menubar.addMenu('&File')
        file.addAction(ofile)
        file.addAction(sep)
        file.addAction(quit)

        edit = menubar.addMenu('&Edit')
        edit.addAction(run)

        help = menubar.addMenu('&Help')

        self.__options_dialog__ = options.optiondialog(self)

    def openfile(self):

        self.__inputfile__ = QtGui.QFileDialog.getOpenFileName(self) 

    def mainrun(self):

        self.__options_dialog__.exec_()
