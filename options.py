from PyQt4 import QtCore, QtGui
import sys

sys.path.append("./module")
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

        self.inftcb = QtGui.QCheckBox("Simulate infinite time", self)
        self.inftcb.setChecked(self.__inftime__)

        labelstep = QtGui.QLabel("Step: ", self)
        self.steptb = QtGui.QLineEdit(str(self.__step__), self)
        self.steptb.move(20, 20)
        self.steptb.resize(280,40)

        labeltprev = QtGui.QLabel("Tprev: ", self)
        self.tprevtb = QtGui.QLineEdit(str(self.__tprev__), self)
        self.tprevtb.move(20, 20)
        self.tprevtb.resize(280,40)

        labelnofrun = QtGui.QLabel("Max num of iterations: ", self)
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
           self.__tprev__ = float(str(self.tprevtb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "TPREV unexpected value reset to defaul")

        if (basicutils.is_integer(str(self.nofruntb.displayText()))):
           self.__nofrun__ = float(str(self.nofruntb.displayText()))
        else:
           QtGui.QMessageBox.critical( self, \
            "ERROR", \
            "Max. num of iterations unexpected value reset to defaul")

        self.__inftime__ = self.inftcb.isChecked()

        self.close()

    def getstep (self):
        return self.__step__

    def gettprev (self):
        return self.__step__

    def getnofrun (self):
        return self.__nofrun__

    def getinftime (self):
        return self.__inftime__

