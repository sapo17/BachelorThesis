import sys

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QGridLayout, QLabel, QPushButton, QWidget


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_index = 0
        self._filenames = []

        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.print_button = QPushButton("Print")
        self.label = QLabel()

        lay = QGridLayout(self)
        lay.addWidget(self.previous_button, 0, 0)
        lay.addWidget(self.next_button, 0, 1)
        lay.addWidget(self.print_button, 0, 2)
        lay.addWidget(self.label, 1, 0, 1, 3)

        self.previous_button.clicked.connect(self.handle_previous)
        self.next_button.clicked.connect(self.handle_next)
        self.print_button.clicked.connect(self.handle_print)

        self._update_button_status(False, True)

        self.load_files()

    def load_files(self):
        self._filenames = [
            "image1.png",
            "image2.png",
            "image3.png",
            "image4.png",
            "image5.png",
            "image6.png",
        ]
        self.current_index = 0

    def handle_next(self):
        self.current_index += 1

    def handle_previous(self):
        self.current_index -= 1

    def handle_print(self):
        for filename in self._filenames:
            print(filename)

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, index):
        if index <= 0:
            self._update_button_status(False, True)
        elif index >= (len(self._filenames) - 1):
            self._update_button_status(True, False)
        else:
            self._update_button_status(True, True)

        if 0 <= index < len(self._filenames):
            self._current_index = index
            filename = self._filenames[self._current_index]
            pixmap = QPixmap(filename)
            self.label.setPixmap(pixmap)

    def _update_button_status(self, previous_enable, next_enable):
        self.previous_button.setEnabled(previous_enable)
        self.next_button.setEnabled(next_enable)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Widget()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec())