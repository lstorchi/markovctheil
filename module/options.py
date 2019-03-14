from PyQt4 import QtCore, QtGui
import sys

import basicutils

class optiondialog_cp(QtGui.QDialog):

    def __init__(self, parent=None):
        self.__numofcp__ = 1
        self.__cp1start__ = 1
        self.__cp1stop__ = -1
        self.__cp2start__ = 1
        self.__cp2stop__ = -1
        self.__cp3start__ = 1
        self.__cp3stop__ = -1
        self.__deltacp__ = 1
        self.__performtest__ = "-1;0"

        super(optiondialog_cp, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        labelnumofcp = QtGui.QLabel("Number of CPs: ", self)
        self.numofcp_line = QtGui.QLineEdit(str(self.__numofcp__), self)
        self.numofcp_line.move(20, 20)
        self.numofcp_line.resize(280,40)

        labelcp1start = QtGui.QLabel("CP1 start: ", self)
        self.cp1start_line = QtGui.QLineEdit(str(self.__cp1start__), self)
        self.cp1start_line.move(20, 20)
        self.cp1start_line.resize(280,40)

        labelcp1stop = QtGui.QLabel("CP1 stop: ", self)
        self.cp1stop_line = QtGui.QLineEdit(str(self.__cp1stop__), self)
        self.cp1stop_line.move(20, 20)
        self.cp1stop_line.resize(280,40)

        labelcp2start = QtGui.QLabel("CP2 start: ", self)
        self.cp2start_line = QtGui.QLineEdit(str(self.__cp2start__), self)
        self.cp2start_line.move(20, 20)
        self.cp2start_line.resize(280,40)

        labelcp2stop = QtGui.QLabel("CP2 stop: ", self)
        self.cp2stop_line = QtGui.QLineEdit(str(self.__cp2stop__), self)
        self.cp2stop_line.move(20, 20)
        self.cp2stop_line.resize(280,40)

        labelcp3start = QtGui.QLabel("CP3 start: ", self)
        self.cp3start_line = QtGui.QLineEdit(str(self.__cp3start__), self)
        self.cp3start_line.move(20, 20)
        self.cp3start_line.resize(280,40)

        labelcp3stop = QtGui.QLabel("CP3 stop: ", self)
        self.cp3stop_line = QtGui.QLineEdit(str(self.__cp3stop__), self)
        self.cp3stop_line.move(20, 20)
        self.cp3stop_line.resize(280,40)

        labeldeltacp = QtGui.QLabel("Delta CP: ", self)
        self.deltacp_line = QtGui.QLineEdit(str(self.__deltacp__), self)
        self.deltacp_line.move(20, 20)
        self.deltacp_line.resize(280,40)

        labelperformtest = QtGui.QLabel("Perfom Lambda test for the specified number of CPs: ", self)
        self.performtest_line = QtGui.QLineEdit(str(self.__performtest__), self)
        self.performtest_line.move(20, 20)
        self.performtest_line.resize(280,40)


        self.grid = QtGui.QGridLayout(self)

        self.grid.addWidget(labelnumofcp)
        self.grid.addWidget(self.numofcp_line)

        self.grid.addWidget(labelcp1start)
        self.grid.addWidget(self.cp1start_line)

        self.grid.addWidget(labelcp1stop)
        self.grid.addWidget(self.cp1stop_line)

        self.grid.addWidget(labelcp2start)
        self.grid.addWidget(self.cp2start_line)

        self.grid.addWidget(labelcp2stop)
        self.grid.addWidget(self.cp2stop_line)

        self.grid.addWidget(labelcp3start)
        self.grid.addWidget(self.cp3start_line)

        self.grid.addWidget(labelcp3stop)
        self.grid.addWidget(self.cp3stop_line)

        self.grid.addWidget(labeldeltacp)
        self.grid.addWidget(self.deltacp_line)

        self.grid.addWidget(labelperformtest)
        self.grid.addWidget(self.performtest_line)

        self.grid.addWidget(self.okbutton)

    def closedialog(self):
        if (basicutils.is_integer(str(self.numofcp_line.displayText()))):
           self.__numofcp__ = int(str(self.numofcp_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "numofcp unexpected value reset to default")

        if (basicutils.is_integer(str(self.cp1start_line.displayText()))):
           self.__cp1start__ = int(str(self.cp1start_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp1start unexpected value reset to default")


        if (basicutils.is_integer(str(self.cp1stop_line.displayText()))):
           self.__cp1stop__ = int(str(self.cp1stop_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp1stop unexpected value reset to default")


        if (basicutils.is_integer(str(self.cp2start_line.displayText()))):
           self.__cp2start__ = int(str(self.cp2start_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp2start unexpected value reset to default")


        if (basicutils.is_integer(str(self.cp2stop_line.displayText()))):
           self.__cp2stop__ = int(str(self.cp2stop_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp2stop unexpected value reset to default")

        if (basicutils.is_integer(str(self.cp3start_line.displayText()))):
           self.__cp3start__ = int(str(self.cp3start_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp3start unexpected value reset to default")


        if (basicutils.is_integer(str(self.cp3stop_line.displayText()))):
           self.__cp3stop__ = int(str(self.cp3stop_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "cp3stop unexpected value reset to default")


        if (basicutils.is_integer(str(self.deltacp_line.displayText()))):
           self.__deltacp__ = int(str(self.deltacp_line.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "deltacp unexpected value reset to default")

        self.__performtest__ = str(self.performtest_line.displayText())

        self.close()

    def get_numofcp (self):
        return self.__numofcp__ 

    def get_cp1start(self):
        return self.__cp1start__ 

    def get_cp1stop(self):
        return self.__cp1stop__

    def get_cp2start(self):
        return self.__cp2start__ 

    def get_cp2stop(self):
        return self.__cp2stop__ 

    def get_cp3start(self):
        return self.__cp3start__ 

    def get_cp3stop(self):
        return self.__cp3stop__ 

    def get_deltacp(self):
        return self.__deltacp__

    def get_performtest(self):
        return self.__performtest__.split(";")


class optiondialog(QtGui.QDialog):

    def __init__(self, parent=None):
        self.__step__ = 0.25
        self.__tprev__ = 37
        self.__nofrun__ = 100
        self.__inftime__ = False
        self.__usecopula__ = False
       
        super(optiondialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        self.inftcb = QtGui.QCheckBox("Simulation using stationary distribution", self)
        self.inftcb.setChecked(self.__inftime__)

        self.copulacb = QtGui.QCheckBox("Simulation using copula for the reward processes", self)
        self.copulacb.setChecked(self.__usecopula__)

        labelstep = QtGui.QLabel("Bin width: ", self)
        self.steptb = QtGui.QLineEdit(str(self.__step__), self)
        self.steptb.move(20, 20)
        self.steptb.resize(280,40)

        labeltprev = QtGui.QLabel("Simulated period: ", self)
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
            "STEP unexpected value reset to default")

        if (basicutils.is_integer(str(self.tprevtb.displayText()))):
           self.__tprev__ = int(str(self.tprevtb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "TPREV unexpected value reset to default")

        if (basicutils.is_integer(str(self.nofruntb.displayText()))):
           self.__nofrun__ = int(str(self.nofruntb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "Max. num of iterations unexpected value reset to default")

        self.__inftime__ = self.inftcb.isChecked()
        self.__usecopula__ = self.copulacb.isChecked()

        self.close()

    def getstep (self):
        return self.__step__

    def gettprev (self):
        return self.__tprev__

    def getnofrun (self):
        return self.__nofrun__

    def getinftime (self):
        return self.__inftime__

    def getusecopula (self):
        return self._usecopula__

class optionnamedialog(QtGui.QDialog):

    def __init__(self, parent=None):
        self.__namerm__ = "ratings"
        self.__nameir__ = "interest_rates"
       
        super(optionnamedialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        labelratingname = QtGui.QLabel("Name of the observed transition matrix: ", self)
        self.ratingnametb = QtGui.QLineEdit(str(self.__namerm__), self)
        self.ratingnametb.move(20, 20)
        self.ratingnametb.resize(280,40)

        labelinterest_ratesname = QtGui.QLabel("Name of the reward matrix: ", self)
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
        self.__namerm__ = str(self.ratingnametb.displayText())
        self.__nameir__ = str(self.interest_ratestb.displayText())
        self.close()

    def getratingname (self):
        return self.__namerm__

    def getiratingname (self):
        return self.__nameir__

