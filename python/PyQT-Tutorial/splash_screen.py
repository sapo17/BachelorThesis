import sys
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt, QTimer

app = QApplication(sys.argv)

lbl = QLabel('<font color=Green size=12><b> Hello World </b></font>')
lbl.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
lbl.show()

QTimer.singleShot(4000, app.quit)

sys.exit(app.exec())