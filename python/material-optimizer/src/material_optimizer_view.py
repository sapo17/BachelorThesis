"""
Author: Can Hasbay
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import QAction
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import mitsuba as mi
from PyQt6 import QtGui, QtWidgets, QtCore
from src.constants import *


class MaterialOptimizerView(QMainWindow):
    """This class contains information of how to represent and display data."""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(
            QtGui.QIcon(IMAGES_DIR_PATH + WINDOW_ICON_FILE_NAME)
        )
        self.show()

    def initUI(self):
        self.initWindowProperties()
        self.statusBar()
        self.initMenu()

    def initWindowProperties(self):
        self.setGeometry(256, 256, 1024, 720)
        self.setWindowTitle(MATERIAL_OPTIMIZER_STRING)

    def initMenu(self):
        # Menu Bar: Import File
        self.importFile = QAction(IMPORT_STRING, self)
        self.importFile.setShortcut(IMPORT_SHORTCUT_STRING)
        self.importFile.setStatusTip(IMPORT_LABEL_STRING)

        # Menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu(FILE_STRING)
        fileMenu.addAction(self.importFile)

    def showFileDialog(
        self, filterStr: str, isMultipleSelectionOk=False
    ) -> list:
        homeDir = str(Path.home())
        if isMultipleSelectionOk:
            return QFileDialog.getOpenFileNames(
                self, IMPORT_FILE_STRING, homeDir, filterStr
            )[0]
        
        # Single selection
        return QFileDialog.getOpenFileName(
            self, IMPORT_FILE_STRING, homeDir, filterStr
        )[0]

    def showInfoMessageBox(self, text):
        msgBox = QMessageBox()
        msgBox.setText(text)
        msgBox.setWindowTitle(INFO_STRING)
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
        msgBox.exec()

    def initCentralWidget(self, sceneParams: dict):
        centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout(centralWidget)

        self.initTopWidget(centralWidget)
        self.initTableContainer(sceneParams, centralWidget)
        self.initBottomContainer(centralWidget)

        self.centralLayout.addWidget(self.topWidgetContainer)
        self.centralLayout.addWidget(self.tableContainer)
        self.centralLayout.addWidget(self.bottomContainer)
        self.setCentralWidget(centralWidget)

    def initTableContainer(self, sceneParams, centralWidget):
        self.tableContainer = QWidget(centralWidget)
        self.tableContainerLayout = QVBoxLayout(self.tableContainer)
        self.table = self.initTable(sceneParams)
        self.tableContainerLayout.addWidget(self.table)

    def initTopWidget(self, centralWidget):
        self.loadRefImgBtn = QPushButton(LOAD_REF_IMG_LABEL)
        self.topWidgetContainer = QWidget(centralWidget)
        self.topWidgetContainerLayout = QHBoxLayout(self.topWidgetContainer)
        self.topWidgetContainerLayout.addWidget(self.loadRefImgBtn)

    def initBottomContainer(self, centralWidget):
        self.bottomContainer = QWidget(centralWidget)
        self.bottomContainerLayout = QVBoxLayout(self.bottomContainer)
        self.initProgessContainer(centralWidget)
        self.initConfigurationContainer()
        self.bottomContainerLayout.addWidget(self.configContainer)
        self.bottomContainerLayout.addWidget(self.progressContainer)

    def initConfigurationContainer(self):
        self.configContainer = QWidget(self.bottomContainer)
        self.configContainerLayout = QVBoxLayout(self.configContainer)

        # min error text input
        self.minErrContainer = QWidget(self.configContainer)
        self.minErrContainerLayout = QHBoxLayout(self.minErrContainer)
        minErrLabel = QLabel(text=COLUMN_LABEL_MINIMUM_ERROR)
        self.minErrLine = QLineEdit()
        self.minErrLine.setText(str(DEFAULT_MIN_ERR))
        self.minErrContainerLayout.addWidget(minErrLabel)
        self.minErrContainerLayout.addWidget(self.minErrLine)

        # samples per pixel dropdown
        self.sppContainer = QWidget(self.configContainer)
        self.sppContainerLayout = QHBoxLayout(self.sppContainer)
        samplesPerPixelLabel = QLabel(text=SPP_DURING_OPT_STRING)
        self.samplesPerPixelBox = QComboBox()
        self.samplesPerPixelBox.addItems(SUPPORTED_SPP_VALUES)
        self.samplesPerPixelBox.setCurrentText(SUPPORTED_SPP_VALUES[2])
        self.sppContainerLayout.addWidget(samplesPerPixelLabel)
        self.sppContainerLayout.addWidget(self.samplesPerPixelBox)

        # loss function dropdown
        self.lossFunctionContainer = QWidget(self.configContainer)
        self.lossFunctionContainerLayout = QHBoxLayout(
            self.lossFunctionContainer
        )
        lossFunctionLabel = QLabel(text=LOSS_FUNCTION_STRING)
        self.lossFunctionBox = QComboBox()
        self.lossFunctionBox.addItems(LOSS_FUNCTION_STRINGS)
        self.lossFunctionContainerLayout.addWidget(lossFunctionLabel)
        self.lossFunctionContainerLayout.addWidget(self.lossFunctionBox)

        # iteration count input
        self.iterationContainer = QWidget(self.configContainer)
        self.iterationContainerLayout = QHBoxLayout(self.iterationContainer)
        iterationCountLabel = QLabel(text=COLUMN_LABEL_ITERATION_COUNT)
        self.iterationCountLine = QLineEdit()
        self.iterationCountLine.setText(str(DEFAULT_ITERATION_COUNT))
        self.iterationContainerLayout.addWidget(iterationCountLabel)
        self.iterationContainerLayout.addWidget(self.iterationCountLine)

        self.configContainerLayout.addWidget(self.minErrContainer)
        self.configContainerLayout.addWidget(self.sppContainer)
        self.configContainerLayout.addWidget(self.lossFunctionContainer)
        self.configContainerLayout.addWidget(self.iterationContainer)

    def initProgessContainer(self, centralWidget):
        self.progressContainer = QWidget(self.bottomContainer)
        self.progressContainerLayout = QHBoxLayout(self.progressContainer)
        self.progressBar = QProgressBar(self.bottomContainer)
        self.optimizeButton = QPushButton(
            START_OPTIMIZATION_STRING, centralWidget
        )
        self.progressContainerLayout.addWidget(self.progressBar)
        self.progressContainerLayout.addWidget(self.optimizeButton)

    def initTable(self, sceneParams: dict):
        firstKey = list(sceneParams)[0]
        columns = sceneParams[firstKey].keys()
        rowsLength = len(sceneParams)
        columnsLength = len(columns)
        result = QTableWidget(rowsLength, columnsLength)
        result.setHorizontalHeaderLabels(columns)
        result.setVerticalHeaderLabels(sceneParams.keys())
        result.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        for row, param in enumerate(sceneParams):
            for col, label in enumerate(columns):
                value = sceneParams[param][label]
                if label == COLUMN_LABEL_VALUE:
                    valueType = type(value)
                    if valueType is mi.Color3f:
                        itemContent = self.Color3fToCellString(value)
                        item = QTableWidgetItem(itemContent)
                    elif valueType is mi.Float:
                        if len(value) == 1:
                            item = QTableWidgetItem(str(value[0]))
                        else:
                            item = QTableWidgetItem(
                                f"mi.Float(length={len(value)})"
                            )
                            item.setFlags(~QtCore.Qt.ItemFlag.ItemIsEditable)
                            item.setBackground(QtGui.QColorConstants.LightGray)
                    elif valueType is mi.TensorXf:
                        item = QTableWidgetItem(str(value))
                        item.setFlags(~QtCore.Qt.ItemFlag.ItemIsEditable)
                        item.setBackground(QtGui.QColorConstants.LightGray)
                    else:
                        item = QTableWidgetItem(NOT_IMPLEMENTED_STRING)
                elif label == COLUMN_LABEL_OPTIMIZE:
                    self.setCheckboxAsQTableItem(result, row, col, value)
                    continue
                else:
                    item = QTableWidgetItem(str(value))
                result.setItem(row, col, item)

        return result

    def setCheckboxAsQTableItem(self, result, row, col, value):
        checkboxContainer = QWidget()
        layout = QHBoxLayout(checkboxContainer)
        item = QCheckBox()
        item.setChecked(value)
        layout.addWidget(item, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        result.setCellWidget(row, col, checkboxContainer)

    @staticmethod
    def Color3fToCellString(color3f: mi.Color3f):
        return str(color3f).translate(str.maketrans({"[": None, "]": None}))

    def replaceTable(self, newTable: QTableWidget):
        self.tableContainerLayout.replaceWidget(self.table, newTable)
        self.table = newTable


class PopUpWindow(QMainWindow):
    def __init__(self, parent: MaterialOptimizerView):
        super(PopUpWindow, self).__init__(parent)
        self.setWindowIcon(
            QtGui.QIcon(IMAGES_DIR_PATH + WINDOW_ICON_FILE_NAME)
        )
        parent.setDisabled(True)
        self.setDisabled(False)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.parent().setDisabled(False)

        # part of it causes dr.jit to warn about leaked variables (specifically
        # sceneParamsHist, lossHist). Setting them to None at least lessens
        # the amount of warned leaked parameters.
        self.lossHist = None
        self.sceneParamsHist = None
        self.initImg = None

        return super().closeEvent(a0)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        super(MplCanvas, self).__init__(fig)
