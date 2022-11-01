# hello.py

"""Simple Hello, World example with PyQt6."""

import sys


# 1. Import QApp and all the required widgets

from PyQt6.QtWidgets import QApplication, QLabel, QWidget


# Create an instance of QApp

app = QApplication([])


# 3. Create your applications GUI
window = QWidget()
window.setWindowTitle('PyQt App')
window.setGeometry(100, 100, 280, 80)
helloMsg = QLabel('<h1>Hello, World!</h1>', parent=window)
helloMsg.move(60, 15)

# 4. show gui
window.show()


# run apps event loop
sys.exit(app.exec())
