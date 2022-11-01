from PyQt6.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton
import sys
import time
 
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(320,250)
        self.setWindowTitle("CodersLegacy")
 
        self.prog_bar = QProgressBar(self)
        self.prog_bar.setGeometry(50, 50, 250, 30)
  
        button = QPushButton("Start", self)
        button.clicked.connect(self.start)
        button.move(50,100)
 
        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.reset)
        reset_button.move(50, 150)
 
    def start(self):
        for val in range(self.prog_bar.maximum()):
            value = self.prog_bar.value()
            self.prog_bar.setValue(value + 1)
            time.sleep(0.2)
 
    def reset(self):
        value = 0
        self.prog_bar.setValue(value)
 
app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())