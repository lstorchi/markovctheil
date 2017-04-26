from PyQt4 import QtGui, QtCore 

import mainmkvcmp
import scipy.io
import options
import numpy

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class main_window(QtGui.QMainWindow):
    
    def __init__(self):
        self.__inputfile__ = ""
        self.__fileio__ = False

        self.__namerm__ = "ratings"
        self.__nameir__ = "interest_rates"
        
        QtGui.QMainWindow.__init__(self) 
        self.resize(640, 480) 
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

        self.__figure__ = plt.figure()
        self.__canvas__ = FigureCanvas(self.__figure__)
        self.setCentralWidget(self.__canvas__)

        self.__options_dialog__ = options.optiondialog(self)

    def openfile(self):

        self.__inputfile__ = QtGui.QFileDialog.getOpenFileName(self) 
        self.__fileio__ = False

        msd = scipy.io.loadmat(str(self.__inputfile__))
        bpd = scipy.io.loadmat(str(self.__inputfile__))
        
        if not(self.__namerm__ in msd.keys()):
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "Cannot find " + self.__namerm__ + " in " + self.__inputfile__)
           return 

        if not(self.__nameir__ in bpd.keys()):
            QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "Cannot find " + self.__nameir__ + " in " + self.__inputfile__)
            return 
        
        if msd[self.__namerm__].shape[0] != bpd[self.__nameir__].shape[0]:
            QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "wrong dim of the input matrix")
            return 
        
        self.__rm__ = msd[self.__namerm__]
        self.__ir__ = bpd[self.__nameir__]
        self.__fileio__ = True

    def mainrun(self):

        if (self.__fileio__):
            self.__options_dialog__.exec_()

            progdialog = QtGui.QProgressDialog(
                    "", "Cancel", 0, 100.0, self)

            progdialog.setWindowTitle("Running")
            progdialog.setWindowModality(QtCore.Qt.WindowModal)
            progdialog.setMinimumDuration(0)
            progdialog.show()
            
            errmsg = []

            tprev = self.__options_dialog__.gettprev()

            entropia = numpy.zeros(tprev, dtype='float64')
            var = numpy.zeros((tprev), dtype='float64')

            if (not mainmkvcmp.main_mkc_comp (self.__rm__, self.__ir__, \
                    self.__options_dialog__.getinftime(), \
                    self.__options_dialog__.getstep(), \
                    self.__options_dialog__.gettprev(), \
                    self.__options_dialog__.getnofrun(), \
                    False, False, False, errmsg, entropia, \
                    var, progdialog)):
                QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    errmsg[0])
                
                progdialog.setValue(100.0)
                progdialog.close()
                
                return 

            progdialog.setValue(100.0)
            progdialog.close()

            self.plot(entropia)

        else:
            QtGui.QMessageBox.critical( self, \
                    "ERROR", \
                    "Error occurs while opening input file")

    def plot(self, data):
        ax = self.__figure__.add_subplot(111)

        ax.hold(False)

        ax.plot(data, '*-')

        self.__canvas__.draw()

    def get_options (self):

        return self.__options_dialog__

