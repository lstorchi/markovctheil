from PyQt4 import QtCore, QtGui

class optiondialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(optiondialog, self).__init__(parent)

        self.okbutton = QtGui.QPushButton('Ok');
        self.okbutton.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold));

        self.connect(self.okbutton, QtCore.SIGNAL("clicked()"), self.closedialog)

        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.okbutton)

    def closedialog(self):
        self.close()
