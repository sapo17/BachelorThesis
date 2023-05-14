"""
Author: Can Hasbay
"""

import datetime
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QAction
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import mitsuba as mi
from PyQt6 import QtGui, QtWidgets, QtCore
from src.constants import *
from enum import Enum
import logging


class PlotStatusEnum(Enum):
    INITIAL = 1
    RENDER = 2
    CLOSE = 3


class MaterialOptimizerView(QMainWindow):
    """This class contains information of how to represent and display data."""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(
            QtGui.QIcon(IMAGES_DIR_PATH + WINDOW_ICON_FILE_NAME)
        )
        self.diffRenderPlot = None
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
        self.tableContainer.setDisabled(True)

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
        self.bottomContainer.setDisabled(True)

    def initConfigurationContainer(self):
        self.configContainer = QWidget(self.bottomContainer)
        self.configContainerLayout = QVBoxLayout(self.configContainer)

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

        # optimization strategy dropdown
        self.optStrategyContainer = QWidget(self.configContainer)
        self.optStrategyContainerLayout = QHBoxLayout(
            self.optStrategyContainer
        )
        optStrategyLabel = QLabel(text=OPTIMIZATION_STRATEGY_LABEL)
        self.optStrategyBox = QComboBox()
        self.optStrategyBox.addItems(OPT_STRATEGY_STRINGS)
        self.optStrategyContainerLayout.addWidget(optStrategyLabel)
        self.optStrategyContainerLayout.addWidget(self.optStrategyBox)

        # advanced settings button
        self.advancedSettingsBtnContainer = QWidget(self.configContainer)
        self.advancedSettingsBtnContainerLayout = QHBoxLayout(
            self.advancedSettingsBtnContainer
        )
        self.advancedSettingsBtn = QPushButton("Advanced Settings")
        self.advancedSettingsBtnContainerLayout.addWidget(
            self.advancedSettingsBtn
        )

        # add to the configuration layout
        self.configContainerLayout.addWidget(self.lossFunctionContainer)
        self.configContainerLayout.addWidget(self.optStrategyContainer)
        self.configContainerLayout.addWidget(self.advancedSettingsBtnContainer)

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
        filteredColumns = list(
            filter(lambda col: col != COLUMN_LABEL_OPTIMIZE, columns)
        )
        rowsLength = len(sceneParams)
        columnsLength = len(filteredColumns)
        result = QTableWidget(rowsLength, columnsLength)
        result.setHorizontalHeaderLabels(filteredColumns)
        result.setVerticalHeaderLabels(sceneParams.keys())
        result.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        for row, param in enumerate(sceneParams):
            for col, label in enumerate(filteredColumns):
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
                else:
                    # special case: mi.Point3f for max/min clamp value
                    isVertexPosOrNormal = VERTEX_POSITIONS_PATTERN.search(
                        param
                    ) or VERTEX_NORMALS_PATTERN.search(param)
                    if isVertexPosOrNormal:
                        if (
                            label == COLUMN_LABEL_MIN_CLAMP_LABEL
                            or label == COLUMN_LABEL_MAX_CLAMP_LABEL
                        ):
                            itemContent = self.Point3fToCellString(value)
                            item = QTableWidgetItem(itemContent)
                        else:
                            item = QTableWidgetItem(str(value))
                    else:
                        item = QTableWidgetItem(str(value))
                result.setStyleSheet(
                    self.getQTableItemSelectedStyle(TOL_BLUE_COLOR)
                )
                result.setItem(row, col, item)

        return result

    @staticmethod
    def getQTableItemSelectedStyle(backgroundColor: str) -> str:
        return (
            "QTableView::item:selected"
            "{"
            f"background-color : {backgroundColor};"
            "selection-color : #000000;"
            "}"
        )

    @staticmethod
    def Color3fToCellString(color3f: mi.Color3f):
        return str(color3f).translate(str.maketrans({"[": None, "]": None}))

    @staticmethod
    def Point3fToCellString(point3f: mi.Point3f):
        return str(point3f).translate(str.maketrans({"[": None, "]": None}))

    def replaceTable(self, newTable: QTableWidget):
        self.tableContainerLayout.replaceWidget(self.table, newTable)
        self.table = newTable

    def showDiffRender(
        self,
        diffRender: mi.Bitmap = None,
        it: int = 0,
        loss: float = 0.0,
        plotStatus: str = CLOSE_STATUS_STR,
        lossHist=[],
        iterationCount=sys.maxsize,
        elapsedTime="",
    ):
        try:
            if PlotStatusEnum[plotStatus] is PlotStatusEnum.CLOSE:
                self.diffRenderPlot.close()
                self.diffRenderPlot = None
            elif PlotStatusEnum[plotStatus] is PlotStatusEnum.INITIAL:
                if diffRender is not None:
                    self.diffRenderPlot = DifferentiableRenderCanvas(
                        diffRender, iterationCount
                    )
                    self.diffRenderPlot.imageAxes.set_axis_off()
                    self.diffRenderPlot.show()

                    # NOTE: uncomment or improve if necessary
                    # self.createDirForDiffRenderFigures()
            elif PlotStatusEnum[plotStatus] is PlotStatusEnum.RENDER:
                if diffRender is not None:
                    self.diffRenderPlot.axesImage.set_data(diffRender)
                    self.diffRenderPlot.imageAxes.set_title(
                        f"Iteration: {it}, Loss: {loss:6f}, Elapsed time: {elapsedTime}",
                        fontsize=14,
                    )
                    self.diffRenderPlot.axesPlot.set_ydata(lossHist)
                    self.diffRenderPlot.axesPlot.set_xdata(
                        np.arange(len(lossHist))
                    )
                    self.diffRenderPlot.plotAxes.relim()
                    self.diffRenderPlot.plotAxes.autoscale_view()
                    self.diffRenderPlot.fig.canvas.draw()
                    self.diffRenderPlot.flush_events()

                    # NOTE: uncomment or improve if necessary
                    # self.outputDiffRenderFig(it)
        except ValueError as err:
            logging.exception(f"Unexpected value in showDiffRender(): {err}")

    def outputDiffRenderFig(self, it):
        outputFileName = self.outputFileDir + f"/{it}.png"
        self.diffRenderPlot.fig.savefig(outputFileName)

    def createDirForDiffRenderFigures(self):
        t = datetime.datetime.now().isoformat("_", "seconds")
        self.outputFileDir = OUTPUT_DIR_PATH + f"diff_render_figures/opt_{t}"
        self.outputFileDir = self.outputFileDir.replace(":", "-")
        Path(self.outputFileDir).mkdir(parents=True, exist_ok=True)

    def getUserDecision(
        self, title, question, defaultButton=QMessageBox.StandardButton.No
    ):
        reply = QMessageBox.question(
            self,
            title,
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton,
        )
        if reply == QMessageBox.StandardButton.Yes:
            return True
        else:
            return False


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
        self.destroy()
        return super().closeEvent(a0)


class AdvancedSettingsPopUp(PopUpWindow):
    def __init__(self, parent: MaterialOptimizerView):
        super(AdvancedSettingsPopUp, self).__init__(parent)
        parent.setDisabled(False)
        self.setWindowTitle("Advanced Settings")

    def initComponents(self):
        """
        Important components are going to be assigned to the parent. These
        are:
            - minErrLine: QLineEdit
            - samplesPerPixelBox: QComboBox
            - iterationCountLine: QLineEdit
            - marginPercentageLine = QLineEdit
            - marginPenalty = QComboBox
        """
        # central widget
        centralWidgetContainer = QWidget()
        centralWidgetContainerLayout = QVBoxLayout(centralWidgetContainer)

        # min error text input
        minErrContainer = QWidget()
        minErrContainerLayout = QHBoxLayout(minErrContainer)
        minErrLabel = QLabel(text=COLUMN_LABEL_MINIMUM_ERROR)
        self.parent().minErrLine = QLineEdit()
        self.parent().minErrLine.setText(str(DEFAULT_MIN_ERR))
        minErrContainerLayout.addWidget(minErrLabel)
        minErrContainerLayout.addWidget(self.parent().minErrLine)

        # samples per pixel dropdown
        sppContainer = QWidget()
        sppContainerLayout = QHBoxLayout(sppContainer)
        samplesPerPixelLabel = QLabel(text=SPP_DURING_OPT_STRING)
        self.parent().samplesPerPixelBox = QComboBox()
        self.parent().samplesPerPixelBox.addItems(SUPPORTED_SPP_VALUES)
        self.parent().samplesPerPixelBox.setCurrentText(
            SUPPORTED_SPP_VALUES[2]
        )
        sppContainerLayout.addWidget(samplesPerPixelLabel)
        sppContainerLayout.addWidget(self.parent().samplesPerPixelBox)

        # iteration count input
        iterationContainer = QWidget()
        iterationContainerLayout = QHBoxLayout(iterationContainer)
        iterationCountLabel = QLabel(text=COLUMN_LABEL_ITERATION_COUNT)
        self.parent().iterationCountLine = QLineEdit()
        self.parent().iterationCountLine.setText(str(DEFAULT_ITERATION_COUNT))
        iterationContainerLayout.addWidget(iterationCountLabel)
        iterationContainerLayout.addWidget(self.parent().iterationCountLine)

        # margin percentage container
        marginPercentageContainer = QWidget()
        marginPercentageContainerLayout = QHBoxLayout(
            marginPercentageContainer
        )
        # margin percentage input
        marginPercentageLabel = QLabel(text=MARGIN_PERCENTAGE_LABEL)
        self.parent().marginPercentageLine = QLineEdit()
        self.parent().marginPercentageLine.setText(INF_STR)
        marginPercentageContainerLayout.addWidget(marginPercentageLabel)
        marginPercentageContainerLayout.addWidget(
            self.parent().marginPercentageLine
        )
        # margin percentage penalty
        penaltyLabel = QLabel(text=MARGIN_PENALTY_LABEL)
        self.parent().marginPenalty = QComboBox()
        self.parent().marginPenalty.addItems([NONE_STR, EXPONENTIAL_DECAY_STR])
        self.parent().marginPenalty.setCurrentText(NONE_STR)
        marginPercentageContainerLayout.addWidget(penaltyLabel)
        marginPercentageContainerLayout.addWidget(self.parent().marginPenalty)

        # use uniform adam (advanced vertex pos. opt.)
        self.parent().useUniformAdamContainer = QWidget()
        uniformAdamLayout = QHBoxLayout(self.parent().useUniformAdamContainer)
        useUniformAdamLabel = QLabel(text=UNIFORM_ADAM_LABEL)
        self.parent().useUniformAdamBox = QComboBox()
        self.parent().useUniformAdamBox.addItems(FALSE_TRUE_IN_STR)
        self.parent().useUniformAdamBox.setCurrentText(FALSE_TRUE_IN_STR[0])
        uniformAdamLayout.addWidget(useUniformAdamLabel)
        uniformAdamLayout.addWidget(self.parent().useUniformAdamBox)
        self.parent().useUniformAdamContainer.setHidden(True)

        # lambda (advanced vertex pos. opt.)
        self.parent().lambdaContainer = QWidget()
        lambdaContainerLayout = QHBoxLayout(self.parent().lambdaContainer)
        lambdaLabel = QLabel(text=LAMBDA_PARAM_MATRIX_LABEL)
        self.parent().lambdaLine = QLineEdit()
        self.parent().lambdaLine.setText(DEFAULT_LAMBDA_PARAM_MATRIX_VALUE)
        lambdaContainerLayout.addWidget(lambdaLabel)
        lambdaContainerLayout.addWidget(self.parent().lambdaLine)
        self.parent().lambdaContainer.setHidden(True)

        # remesh step size (advanced vertex pos. opt.)
        self.parent().remeshContainer = QWidget()
        remeshContainerLayout = QHBoxLayout(self.parent().remeshContainer)
        remeshStepSizeLabel = QLabel(text=REMESH_STEP_LABEL)
        self.parent().remeshBox = QComboBox()
        self.parent().remeshBox.addItems(REMESH_DROPDOWN_VALUES)
        self.parent().remeshBox.setCurrentText(NONE_STR)
        remeshContainerLayout.addWidget(remeshStepSizeLabel)
        remeshContainerLayout.addWidget(self.parent().remeshBox)
        self.parent().remeshContainer.setHidden(True)

        centralWidgetContainerLayout.addWidget(minErrContainer)
        centralWidgetContainerLayout.addWidget(sppContainer)
        centralWidgetContainerLayout.addWidget(iterationContainer)
        centralWidgetContainerLayout.addWidget(marginPercentageContainer)
        centralWidgetContainerLayout.addWidget(
            self.parent().useUniformAdamContainer
        )
        centralWidgetContainerLayout.addWidget(self.parent().lambdaContainer)
        centralWidgetContainerLayout.addWidget(self.parent().remeshContainer)
        self.setCentralWidget(centralWidgetContainer)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        return  # nothing to do here...


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig, self.axes = plt.subplots(1, 5, figsize=(20, 4))
        super(MplCanvas, self).__init__(fig)


class DifferentiableRenderCanvas(FigureCanvasQTAgg):
    def __init__(self, diffRender, iterationCount):
        self.fig, (self.imageAxes, self.plotAxes) = plt.subplots(
            2, 1, height_ratios=[5, 1]
        )
        plt.subplots_adjust(hspace=0.025, right=0.975, top=0.95)
        plt.autoscale(tight=True)
        self.plotAxes.set_xlabel(ITERATION_STRING, fontsize=14)
        self.plotAxes.set_ylabel(LOSS_STRING, fontsize=14)
        super(DifferentiableRenderCanvas, self).__init__(self.fig)
        self.axesImage = self.imageAxes.imshow(diffRender, aspect="auto")
        (self.axesPlot,) = self.plotAxes.plot([], [])
        self.plotAxes.set_xlim([0, iterationCount])
