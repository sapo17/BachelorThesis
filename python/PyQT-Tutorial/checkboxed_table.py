import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt6.QtCore import Qt


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.windowWidth, self.windowHeight = 512, 256
        self.resize(self.windowWidth, self.windowHeight)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.btn = QPushButton('&Ret vals')
        self.btn.clicked.connect(self.retrieveCheckboxVals)
        self.layout.addWidget(self.btn)

        self.table = QTableWidget(3, 3)
        self.layout.addWidget(self.table)

        for row in range(3):
            for col in range(3):
                item = QTableWidgetItem('Item {0}-{1}'.format(row, col))
                if col % 3 == 0:
                    item.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                                  Qt.ItemFlag.ItemIsEnabled)
                    item.setCheckState(Qt.CheckState.Checked)
                    self.table.setItem(row, col, item)
                else:
                    self.table.setItem(row, col, item)


    def retrieveCheckboxVals(self):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                print([self.table.item(row, col).text()
                      for col in range(self.table.columnCount())])

    def toggleRow(self, row):
        print('test')
        # self.table.item(0, 0).setFlags(Qt.ItemFlag.ItemIsUserCheckable |
        #                           Qt.ItemFlag.ItemIsEnabled)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')
