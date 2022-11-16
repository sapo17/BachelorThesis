from PyQt6.QtWidgets import QComboBox, QMainWindow, QApplication, QWidget
import sys



def activated(index):
    print("Activated index:", index)

def text_changed(s):
    print("Text changed:", s)

def index_changed(index):
    print("Index changed", index)

app = QApplication(sys.argv)
w = QWidget()
w.setGeometry(100, 100, 280, 80)
combobox = QComboBox(w)
combobox.addItems(['One', 'Two', 'Three', 'Four'])

# Connect signals to the methods.
combobox.activated.connect(activated)
combobox.currentTextChanged.connect(text_changed)
combobox.currentIndexChanged.connect(index_changed)
w.show()

app.exec()