# signals_slots.py

"""Signals and slots example."""

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

def greet(name):
    msgLabel.setText(f"Hello, {name}")

app = QApplication([])
window = QWidget()
window.setWindowTitle("Signals and slots")
layout = QVBoxLayout()

button = QPushButton("Greet")
button.clicked.connect(lambda: greet("Sapo!") )

layout.addWidget(button)
msgLabel = QLabel("")
layout.addWidget(msgLabel)
window.setLayout(layout)
window.show()
sys.exit(app.exec())