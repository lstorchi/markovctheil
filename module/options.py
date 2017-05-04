from PyQt4 import QtCore, QtGui
import sys

import basicutils

class optiondialog(QtGui.QDialog):

    def __init__(self, parent=None):
        self.__step__ = 0.25
        self.__tprev__ = 37
        self.__nofrun__ = 100
        self.__inftime__ = False
       
        super(optiondialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        self.inftcb = QtGui.QCheckBox("Simulation using stationary distribution", self)
        self.inftcb.setChecked(self.__inftime__)

        labelstep = QtGui.QLabel("Bin width: ", self)
        self.steptb = QtGui.QLineEdit(str(self.__step__), self)
        self.steptb.move(20, 20)
        self.steptb.resize(280,40)

        labeltprev = QtGui.QLabel("Forecasted period: ", self)
        self.tprevtb = QtGui.QLineEdit(str(self.__tprev__), self)
        self.tprevtb.move(20, 20)
        self.tprevtb.resize(280,40)

        labelnofrun = QtGui.QLabel("Monte Carlo iterations: ", self)
        self.nofruntb = QtGui.QLineEdit(str(self.__nofrun__), self)
        self.nofruntb.move(20, 20)
        self.nofruntb.resize(280,40)

        self.grid = QtGui.QGridLayout(self)

        self.grid.addWidget(self.inftcb)

        self.grid.addWidget(labelstep)
        self.grid.addWidget(self.steptb)

        self.grid.addWidget(labeltprev)
        self.grid.addWidget(self.tprevtb)

        self.grid.addWidget(labelnofrun)
        self.grid.addWidget(self.nofruntb)

        self.grid.addWidget(self.okbutton)

    def closedialog(self):
        if (basicutils.is_float(str(self.steptb.displayText()))):
           self.__step__ = float(str(self.steptb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "STEP unexpected value reset to defaul")

        if (basicutils.is_integer(str(self.tprevtb.displayText()))):
           self.__tprev__ = int(str(self.tprevtb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "TPREV unexpected value reset to defaul")

        if (basicutils.is_integer(str(self.nofruntb.displayText()))):
           self.__nofrun__ = int(str(self.nofruntb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "Max. num of iterations unexpected value reset to defaul")

        self.__inftime__ = self.inftcb.isChecked()

        self.close()

    def getstep (self):
        return self.__step__

    def gettprev (self):
        return self.__tprev__

    def getnofrun (self):
        return self.__nofrun__

    def getinftime (self):
        return self.__inftime__

class optionnamedialog(QtGui.QDialog):

    def __init__(self, parent=None):
        self.__namerm__ = "ratings"
        self.__nameir__ = "interest_rates"
       
        super(optionnamedialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        labelratingname = QtGui.QLabel("Name of the transition probability matrix: ", self)
        self.ratingnametb = QtGui.QLineEdit(str(self.__namerm__), self)
        self.ratingnametb.move(20, 20)
        self.ratingnametb.resize(280,40)

        labelinterest_ratesname = QtGui.QLabel("Name of the rewards matrix: ", self)
        self.interest_ratestb = QtGui.QLineEdit(str(self.__nameir__), self)
        self.interest_ratestb.move(20, 20)
        self.interest_ratestb.resize(280,40)

        self.grid = QtGui.QGridLayout(self)

        self.grid.addWidget(labelratingname)
        self.grid.addWidget(self.ratingnametb)

        self.grid.addWidget(labelinterest_ratesname)
        self.grid.addWidget(self.interest_ratestb)

        self.grid.addWidget(self.okbutton)

    def closedialog(self):
        self.close()

    def getratingname (self):
        return self.__namerm__

    def getiratingname (self):
        return self.__nameir__

