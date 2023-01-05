import datetime
import json
import sys
import mitsuba as mi
import drjit as dr
import logging
from PyQt6.QtWidgets import *

import numpy as np
from src.constants import *
from src.material_optimizer_model import MaterialOptimizerModel
from src.material_optimizer_view import (
    MaterialOptimizerView,
    MplCanvas,
    PopUpWindow,
)


class MaterialOptimizerController:
    """This class implements the commands for the model (data) or view (data representation)."""

    def __init__(
        self, model: MaterialOptimizerModel, view: MaterialOptimizerView
    ):
        self.model = model
        self.view = view
        tableValues = self.combineTableValues(
            self.model.initialSceneParams, self.model.optimizationParams
        )
        self.view.initCentralWidget(tableValues)
        self.connectSignals()

    def loadMitsubaScene(self):
        try:
            fileName = self.view.showFileDialog(XML_FILE_FILTER_STRING)
            self.model.loadMitsubaScene(fileName)
            self.model.resetReferenceImage()
            self.model.setSceneParams(self.model.scene)
            self.model.setInitialSceneParams(self.model.sceneParams)
            self.model.setDefaultOptimizationParams(
                self.model.initialSceneParams
            )
            self.updateTable(
                self.model.initialSceneParams, self.model.optimizationParams
            )
        except Exception as err:
            self.model.loadMitsubaScene()
            msg = "Invalid Mitsuba 3 scene file. Setting default scene."
            msg += " Please make sure that the loaded scene file contains "
            msg += " the default parameters 'integrator', 'resx', 'resy'. "
            msg += " Please refer to 'scenes\cbox.xml' for an example."
            msg += f" Mitsuba Error: {err=}"
            self.view.showInfoMessageBox(msg)

    def updateTable(self, params, optimizationParams):
        newTable = self.view.initTable(
            self.combineTableValues(params, optimizationParams)
        )
        self.view.replaceTable(newTable)
        self.updateSignals()

    def loadReferenceImage(self):
        try:
            refImgFileName = self.view.showFileDialog(
                IMAGES_FILE_FILTER_STRING
            )
            readImg = self.model.readImage(refImgFileName)
            self.model.refImage = readImg
        except Exception as err:
            logging.error(str(err))
            self.view.showInfoMessageBox("Cannot load the reference image.")

    def updateSignals(self):
        self.view.table.cellChanged.connect(self.onCellChanged)

    def combineTableValues(self, params, optimizationParams):
        result = {}
        for key in params:
            result[key] = {COLUMN_LABEL_VALUE: params[key]}
            result[key].update(optimizationParams[key])
        return result

    def onOptimizeBtnClicked(self):
        try:
            self.optimizeMaterials()
        except Exception as err:
            msg = f"Exiting program. Runtime error during optimiztion: {err}"
            logging.error(msg)
            self.view.showInfoMessageBox(msg)
            sys.exit()

    def optimizeMaterials(self):

        # Precondition: (1) reference image is loaded, and
        #               (2) at least one checked scene parameter
        checkedRows = self.getCheckedRows()
        if self.model.refImage is None or len(checkedRows) <= 0:
            msg = "Please make sure to load a reference image and to check at least one scene parameter for the optimization."
            self.view.showInfoMessageBox(msg)
            return

        # If the optimization being restarted: refresh progress bar and,
        # update scene parameters according to the view
        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()
            currentSceneParams = self.getCurrentSceneParams()
            self.model.updateSceneParameters(currentSceneParams)

        self.view.optimizeButton.setDisabled(True)
        self.view.progressBar.setValue(50)

        opts, initImg = self.model.prepareOptimization(checkedRows)

        self.view.progressBar.setValue(100)
        self.view.progressBar.reset()

        # initiate the optimization loop
        lossHist, sceneParamsHist = self.model.optimizationLoop(
            opts, lambda x: self.view.progressBar.setValue(x)
        )

        self.view.progressBar.setValue(self.view.progressBar.maximum())
        logging.info(f"Initial scene parameters:\n {sceneParamsHist[0]}")

        if len(lossHist) <= 1:
            self.view.showInfoMessageBox("No optimization was necessary")
        else:
            popUp = PopUpWindow(self.view)
            self.initOptimizedSceneSelector(
                popUp, initImg, lossHist, sceneParamsHist
            )

        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText(RESTART_OPTIMIZATION_STRING)

    def getCurrentSceneParams(self):
        currentSceneParams = {}
        for row in range(self.view.table.rowCount()):
            key = self.view.table.verticalHeaderItem(row).text()
            initValue = self.model.initialSceneParams[key]
            newValue = self.view.table.item(row, 0).text()
            valueType = type(initValue)
            if valueType is mi.Color3f:
                currentSceneParams[key] = self.model.stringToColor3f(newValue)
            elif valueType is mi.Float:
                if len(initValue) == 1:
                    currentSceneParams[key] = mi.Float(float(newValue))
            elif valueType is mi.TensorXf:
                currentSceneParams[key] = dr.zeros(
                    mi.TensorXf, initValue.shape
                )

        return currentSceneParams

    def connectSignals(self):
        self.view.importFile.triggered.connect(self.loadMitsubaScene)
        self.view.optimizeButton.clicked.connect(self.onOptimizeBtnClicked)
        self.view.table.cellChanged.connect(self.onCellChanged)
        self.view.loadRefImgBtn.clicked.connect(
            self.onLoadReferenceImageBtnClicked
        )
        self.view.minErrLine.editingFinished.connect(self.onMinErrLineChanged)
        self.view.iterationCountLine.editingFinished.connect(
            self.onIterationCountLineChanged
        )
        self.view.samplesPerPixelBox.currentTextChanged.connect(
            self.onSamplesPerPixelChanged
        )
        self.view.lossFunctionBox.currentTextChanged.connect(
            self.onLossFunctionChanged
        )

    def onMinErrLineChanged(self):
        try:
            self.model.setMinError(self.view.minErrLine.text())
        except Exception as err:
            self.view.minErrLine.setText(str(DEFAULT_MIN_ERR))
            self.view.showInfoMessageBox(str(err))

    def onIterationCountLineChanged(self):
        try:
            self.model.setIterationCount(self.view.iterationCountLine.text())
        except Exception as err:
            self.view.iterationCountLine.setText(str(DEFAULT_ITERATION_COUNT))
            self.view.showInfoMessageBox(str(err))

    def onSamplesPerPixelChanged(self, text: str):
        try:
            self.model.setSamplesPerPixel(text)
            if self.model.samplesPerPixel > 16:
                msg = "Beware that higher samples per pixel rate during "
                msg += "optimization may cause to unwanted crashes because of "
                msg += "GPU Memory(CUDA) limitations."
                self.view.showInfoMessageBox(msg)
        except Exception as err:
            self.view.samplesPerPixelBox.setCurrentText(
                SUPPORTED_SPP_VALUES[0]
            )
            self.view.showInfoMessageBox(str(err))

    def onLossFunctionChanged(self, text: str):
        self.model.setLossFunction(text)

    def onLoadReferenceImageBtnClicked(self):
        self.model.setMinError(self.view.minErrLine.text())
        self.model.setIterationCount(self.view.iterationCountLine.text())
        self.loadReferenceImage()

    def getColumnIndex(self, columnLabel: str):
        for colIdx in range(self.view.table.columnCount()):
            if (
                self.view.table.horizontalHeaderItem(colIdx).text()
                == columnLabel
            ):
                return colIdx
        return 0

    def getCheckedRows(self) -> list:
        result = []
        colIdx = self.getColumnIndex(COLUMN_LABEL_OPTIMIZE)
        for rowIdx in range(self.view.table.rowCount()):
            assert (
                self.view.table.cellWidget(rowIdx, colIdx).layout().itemAt(0)
                != None
            )
            isChecked = (
                self.view.table.cellWidget(rowIdx, colIdx)
                .layout()
                .itemAt(0)
                .widget()
                .isChecked()
            )
            if isChecked:
                result.append(
                    self.view.table.verticalHeaderItem(rowIdx).text()
                )
        return result

    def onCellChanged(self, row, col):
        if col == 0:
            isSuccess, param, newValue = self.onSceneParamChanged(row, col)
            if isSuccess:
                self.model.updateSceneParam(param, newValue)
        else:
            (
                isSuccess,
                paramRow,
                paramCol,
                newValue,
            ) = self.onOptimizationParamChanged(row, col)
            if isSuccess:
                self.model.updateOptimizationParam(
                    paramCol, paramRow, newValue
                )

    def onOptimizationParamChanged(self, row, col):
        paramRow = self.view.table.verticalHeaderItem(
            self.view.table.currentRow()
        ).text()
        paramCol = self.view.table.horizontalHeaderItem(
            self.view.table.currentColumn()
        ).text()
        try:
            newValue = float(self.view.table.item(row, col).text())
        except:
            self.view.table.item(row, col).setText(
                str(self.model.optimizationParams[paramRow][paramCol])
            )
            errMsg = "Optimization parameter is not changed."
            self.view.showInfoMessageBox(errMsg)
            return False, None, None, None
        return True, paramRow, paramCol, newValue

    def onSceneParamChanged(self, row, col):
        param = self.view.table.verticalHeaderItem(
            self.view.table.currentRow()
        ).text()
        newValue = self.view.table.item(row, col).text()
        paramType = type(self.model.sceneParams[param])
        if paramType is mi.Color3f:
            try:
                newValue = self.model.stringToColor3f(newValue)
            except ValueError as err:
                self.view.table.item(row, col).setText(
                    MaterialOptimizerView.Color3fToCellString(
                        self.model.initialSceneParams[param]
                    )
                )
                errMsg = "Scene parameter is not changed. " + str(err)
                self.view.showInfoMessageBox(errMsg)
                return False, None, None
        elif paramType is mi.Float:
            if len(
                self.model.initialSceneParams[param]
            ) != 1 or not MaterialOptimizerModel.is_float(newValue):
                self.view.table.item(row, col).setText(
                    str(self.model.initialSceneParams[param][0])
                )
                errMsg = "Scene parameter is not changed."
                errMsg += " Please provide a valid float value (e.g. 0.1)."
                self.view.showInfoMessageBox(errMsg)
                return False, None, None
            newValue = mi.Float(float(newValue))
        return True, param, newValue

    def initOptimizedSceneSelector(
        self, popUpWindow: PopUpWindow, initImg, lossHist, sceneParamsHist
    ):
        popUpWindow.initImg = initImg
        popUpWindow.lossHist = lossHist
        popUpWindow.sceneParamsHist = sceneParamsHist

        # central widget
        centralWidgetContainer = QWidget()
        centralWidgetContainerLayout = QVBoxLayout(centralWidgetContainer)

        # dropdown menu
        popUpWindow.comboBox = QComboBox()
        popUpWindow.comboBox.addItems(
            [LAST_ITERATION_STRING, COLUMN_LABEL_MINIMUM_ERROR]
        )
        popUpWindow.comboBox.currentTextChanged.connect(
            lambda: self.onOptimizedSceneSelectorTextChanged(popUpWindow)
        )

        # output button
        outputBtn = QPushButton(text=OUTPUT_TO_JSON_STRING)
        outputBtn.clicked.connect(lambda: self.onOutputBtnPressed(popUpWindow))

        centralWidgetContainerLayout.addWidget(popUpWindow.comboBox)
        centralWidgetContainerLayout.addWidget(outputBtn)
        popUpWindow.setCentralWidget(centralWidgetContainer)

        popUpWindow.show()

        lastIteration = len(popUpWindow.sceneParamsHist) - 1
        self.showOptimizedPlot(popUpWindow, lastIteration)

    def onOutputBtnPressed(self, popUpWindow: PopUpWindow):
        selectedIteration = len(popUpWindow.sceneParamsHist) - 1
        if popUpWindow.comboBox.currentText() == COLUMN_LABEL_MINIMUM_ERROR:
            selectedIteration = MaterialOptimizerModel.minIdxInDrList(
                popUpWindow.lossHist
            )

        outputFileDir = (
            OUTPUT_DIR_PATH
            + datetime.datetime.now().isoformat("_", "seconds")
            + "_iteration_"
            + str(selectedIteration)
        )
        outputFileDir = outputFileDir.replace(":", "-")
        Path(outputFileDir).mkdir(parents=True, exist_ok=True)
        outputFileName = outputFileDir + "/optimized_params.json"

        # fill the dictionary with appropriate parameter values and output
        # if scene parameter is special (e.g. bitmap texture) then output it in
        # another file (e.g. bitmap texture: output .png), otherwise fill it in
        # the dictionary and finally output as a .json file
        with open(outputFileName, "w") as outfile:
            outputDict = {}
            for k, v in popUpWindow.sceneParamsHist[selectedIteration].items():
                if type(v) is mi.TensorXf:
                    # special output: volume
                    if ALBEDO_DATA_PATTERN.search(k):
                        outputVolumeFileName = (
                            outputFileDir + f"//optimized_volume_{k}.vol"
                        )
                        mi.VolumeGrid(v).write(outputVolumeFileName)
                    else:
                        # special output: bitmap texture
                        outputTextureFileName = (
                            outputFileDir + f"/optimized_texture_{k}.png"
                        )
                        mi.util.write_bitmap(outputTextureFileName, v)
                elif type(v) is mi.Float:
                    floatArray = [f for f in v]
                    if VERTEX_COLOR_PATTERN.search(k):
                        # special output: vertex colors as numpy array
                        outputVertexColorFileName = (
                            outputFileDir
                            + f"/optimized_vertex_color_array_{k}.npy"
                        )
                        np.save(
                            outputVertexColorFileName, np.array(floatArray)
                        )
                    else:
                        # otherwise: fill the dictionary
                        outputDict[k] = floatArray
                else:
                    # default case: fill the dictionary
                    outputDict[k] = str(v)

            # output: filled dictionary
            json.dump(outputDict, outfile, indent=4)

            # inform user
            absPath = str(Path(outputFileDir).resolve())
            self.view.showInfoMessageBox(
                f"The output can be found at: '{absPath}'"
            )

    def onOptimizedSceneSelectorTextChanged(self, popUpWindow: PopUpWindow):
        if popUpWindow.comboBox.currentText() == COLUMN_LABEL_MINIMUM_ERROR:
            self.showOptimizedPlot(
                popUpWindow,
                MaterialOptimizerModel.minIdxInDrList(popUpWindow.lossHist),
            )
        else:
            popUpWindow.showOptimizedPlot(len(popUpWindow.sceneParamsHist) - 1)

    def showOptimizedPlot(self, popUpWindow: PopUpWindow, iteration: int):
        logging.info(
            f"Scene parameters in {iteration}:\n {popUpWindow.sceneParamsHist[iteration]}"
        )
        self.model.sceneParams.update(
            values=popUpWindow.sceneParamsHist[iteration]
        )
        sc = MplCanvas()
        self.plotOptimizationResults(
            sc,
            self.model.refImage,
            popUpWindow.initImg,
            MaterialOptimizerModel.convertToBitmap(
                MaterialOptimizerModel.render(self.model.scene, 512)
            ),
            {self.model.lossFunction: popUpWindow.lossHist},
            iteration,
            popUpWindow.lossHist[iteration][0],
        )

    def plotOptimizationResults(
        self,
        canvas: MplCanvas,
        refImage,
        initImg,
        finalImg,
        paramErrors,
        iterationNumber: int,
        loss: float,
    ):

        for k, v in paramErrors.items():
            canvas.axes[0][0].plot(v, label=k)

        canvas.axes[0][0].set_xlabel(ITERATION_STRING)
        canvas.axes[0][0].set_ylabel(LOSS_STRING)
        canvas.axes[0][0].legend()
        canvas.axes[0][0].set_title(PARAMETER_ERROR_PLOT_STRING)

        canvas.axes[0][1].imshow(
            MaterialOptimizerModel.convertToBitmap(initImg)
        )
        canvas.axes[0][1].axis(OFF_STRING)
        canvas.axes[0][1].set_title(INITIAL_IMAGE_STRING)

        canvas.axes[1][0].imshow(
            MaterialOptimizerModel.convertToBitmap(finalImg)
        )
        canvas.axes[1][0].axis(OFF_STRING)
        canvas.axes[1][0].set_title(
            f"Optimized image: Iteration #{iterationNumber}, Loss: {loss:6f}"
        )

        canvas.axes[1][1].imshow(
            MaterialOptimizerModel.convertToBitmap(refImage)
        )
        canvas.axes[1][1].axis(OFF_STRING)
        canvas.axes[1][1].set_title(REFERENCE_IMAGE_STRING)

        import matplotlib.pyplot

        matplotlib.pyplot.savefig(IMAGES_DIR_PATH + FIGURE_FILE_NAME)

        canvas.show()
