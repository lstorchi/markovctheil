import sys
from PyQt4 import QtGui

import mainwin

def main():
    app = QtGui.QApplication(sys.argv)
    main = mainwin.main_window()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
