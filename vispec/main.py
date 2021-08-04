from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

def window():
	app = QApplication(sys.argv)
	win = QMainWindow()
	win.setGeometry(100, 100, 300, 500)
	win.setWindowTitle("ViSpec")

	label = QtWidgets.QLabel(win)
	label.setText("File name")
	label.move(20,20)

	win.show()
	sys.exit(app.exec_())

window()