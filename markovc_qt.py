import sys
from PyQt5 import QtWidgets

sys.path.append("./module")
import mainwin

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = mainwin.main_window()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
