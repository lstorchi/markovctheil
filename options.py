from PyQt4 import QtCore, QtGui

class optiondialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(optiondialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));
        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        self.inftcb = QtGui.QCheckBox("Simulate infinite time", self)
        self.inftcb.setChecked(False)

        self.grid = QtGui.QGridLayout(self)

        self.grid.addWidget(self.inftcb)
        self.grid.addWidget(self.okbutton)

    def closedialog(self):
        self.close()
