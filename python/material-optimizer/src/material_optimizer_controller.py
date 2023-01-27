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
            self.model.resetSensorToReferenceImageDict()
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

    def loadReferenceImages(self):
        try:
            refImgFileNames = self.view.showFileDialog(
                IMAGES_FILE_FILTER_STRING, isMultipleSelectionOk=True
            )

            # check length of reference images is equal to # of sensors
            if len(refImgFileNames) != len(self.model.scene.sensors()):
                err = "The amount of selected reference images do not match "
                err += "with the available sensors loaded in the scene file."
                raise RuntimeError(err)

            readImgs = [
                self.model.readImage(refImgFileName)
                for refImgFileName in refImgFileNames
            ]

            # Create a dictionary where key: sensor, value: refImage
            self.model.setSensorToReferenceImageDict(readImgs)

            msg = (
                "Reference image/s loaded successfully. Assuming the order of "
            )
            msg += " the loaded images corresponds to the appropriate sensors "
            msg += "defined in the loaded scene file (i.e. first image "
            msg += "corresponds to the first sensor in the loaded scene)."
            self.view.showInfoMessageBox(msg)
        except Exception as err:
            logging.error(str(err))
            self.view.showInfoMessageBox(
                f"Cannot load reference image/s. {str(err)}"
            )

    def updateSignals(self):
        self.view.table.cellChanged.connect(self.onCellChanged)

    def combineTableValues(self, params, optimizationParams):
        result = {}
        for key in params:
            self.updateOptimizationParameters(optimizationParams, key)
            result[key] = {COLUMN_LABEL_VALUE: params[key]}
            result[key].update(optimizationParams[key])
        return result

    def updateOptimizationParameters(self, optimizationParams, key):
        # update min and max clamp values to the key's clamp value
        # for example, if key contains .reflectance string then it will
        # be most likely a REFLECTANCE_PATTERN, thus the corresponding clamp
        # value will be assigned to the optimization parameters
        mostLikelyPattern: re.Pattern = self.model.getClosestPattern(key)
        optimizationParams[key][
            COLUMN_LABEL_MIN_CLAMP_LABEL
        ] = DEFAULT_CLAMP_VALUES[mostLikelyPattern][0]
        optimizationParams[key][
            COLUMN_LABEL_MAX_CLAMP_LABEL
        ] = DEFAULT_CLAMP_VALUES[mostLikelyPattern][1]

    def onOptimizeBtnClicked(self):
        try:
            self.optimizeMaterials()
        except Exception as err:
            msg = f"Exiting program. Runtime error during optimiztion: {err}"
            logging.error(msg)
            self.view.showInfoMessageBox(msg)
            sys.exit()

    def optimizeMaterials(self):

        # Precondition: (1) reference image/s is/are loaded, and
        #               (2) at least one checked scene parameter
        checkedRows = self.getCheckedRows()
        if (
            self.model.sensorToReferenceImageDict is None
            or len(checkedRows) <= 0
        ):
            msg = "Please make sure to load a reference image/s and to check "
            msg += "at least one scene parameter for the optimization."
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
        self.loadReferenceImages()

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
        if col == self.getColumnIndex(COLUMN_LABEL_VALUE):
            # value has changed
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

    def isLegalClampValue(
        self,
        colIdx: int,
        rowIdx: int,
        paramCol: str,
        paramRow: str,
        newValue: float,
    ):
        mostLikelyPattern = self.model.getClosestPattern(paramRow)
        if (
            paramCol == COLUMN_LABEL_MIN_CLAMP_LABEL
            or paramCol == COLUMN_LABEL_MAX_CLAMP_LABEL
        ):
            if paramCol == COLUMN_LABEL_MIN_CLAMP_LABEL:
                if newValue < DEFAULT_CLAMP_VALUES[mostLikelyPattern][0]:
                    msg = "Optimization parameter is not changed. Please make"
                    msg += " sure that the clamp value is not smaller than"
                    msg += f" '{DEFAULT_CLAMP_VALUES[mostLikelyPattern][0]}'."
                    self.view.showInfoMessageBox(msg)
                    self.view.table.item(rowIdx, colIdx).setText(
                        str(DEFAULT_CLAMP_VALUES[mostLikelyPattern][0])
                    )
                    return False
            elif paramCol == COLUMN_LABEL_MAX_CLAMP_LABEL:
                if newValue > DEFAULT_CLAMP_VALUES[mostLikelyPattern][1]:
                    msg = "Optimization parameter is not changed. Please make"
                    msg += " sure that the clamp value is not larger than"
                    msg += f" '{DEFAULT_CLAMP_VALUES[mostLikelyPattern][1]}'."
                    self.view.showInfoMessageBox(msg)
                    self.view.table.item(rowIdx, colIdx).setText(
                        str(DEFAULT_CLAMP_VALUES[mostLikelyPattern][1])
                    )
                    return False

        return True

    def isLegalLearningRate(
        self, row: int, col: int, paramCol: str, newValue: float
    ):
        if paramCol == COLUMN_LABEL_LEARNING_RATE:
            if newValue > MAX_LEARNING_RATE:
                msg = "Optimization parameter is not changed. Please make"
                msg += " sure that the learning rate is not larger than"
                msg += f" '{MAX_LEARNING_RATE}'."
                self.view.showInfoMessageBox(msg)
                self.view.table.item(row, col).setText(str(MAX_LEARNING_RATE))
                return False
            elif newValue < MIN_LEARNING_RATE:
                msg = "Optimization parameter is not changed. Please make"
                msg += " sure that the learning rate is not smaller than"
                msg += f" '{MIN_LEARNING_RATE}'."
                self.view.showInfoMessageBox(msg)
                self.view.table.item(row, col).setText(str(MIN_LEARNING_RATE))
                return False

        return True

    def onOptimizationParamChanged(self, row, col):
        paramRow = self.getRowLabelText()
        paramCol = self.getColumnLabelText()
        try:
            newValue = float(self.view.table.item(row, col).text())

            if not self.isLegalLearningRate(row, col, paramCol, newValue):
                return False, None, None, None
            if not self.isLegalClampValue(
                col, row, paramCol, paramRow, newValue
            ):
                return False, None, None, None
        except:
            self.view.table.item(row, col).setText(
                str(self.model.optimizationParams[paramRow][paramCol])
            )
            errMsg = "Optimization parameter is not changed."
            self.view.showInfoMessageBox(errMsg)
            return False, None, None, None
        return True, paramRow, paramCol, newValue

    def getColumnLabelText(self):
        return self.view.table.horizontalHeaderItem(
            self.view.table.currentColumn()
        ).text()

    def onSceneParamChanged(self, row, col):
        param = self.getRowLabelText()
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

    def getRowLabelText(self) -> str:
        # returns scene parameter key
        return self.view.table.verticalHeaderItem(
            self.view.table.currentRow()
        ).text()

    def initOptimizedSceneSelector(
        self, popUpWindow: PopUpWindow, initImg, lossHist, sceneParamsHist
    ):
        popUpWindow.initImg = initImg
        popUpWindow.lossHist = lossHist
        popUpWindow.sceneParamsHist = sceneParamsHist

        # central widget
        centralWidgetContainer = QWidget()
        centralWidgetContainerLayout = QVBoxLayout(centralWidgetContainer)

        # dropdown menu: sensor selection
        sensorSelectionContainer = QWidget()
        sensorSelectionLabel = QLabel(SENSOR_IDX_LABEL)
        sensorSelectionContainerLayout = QHBoxLayout(sensorSelectionContainer)
        popUpWindow.sensorDropdown = QComboBox()
        popUpWindow.sensorDropdown.addItems(
            [
                str(sensorIdx)
                for sensorIdx, sensor in enumerate(self.model.scene.sensors())
            ]
        )
        popUpWindow.sensorDropdown.currentTextChanged.connect(
            lambda: self.onSensorIdxChanged(popUpWindow)
        )
        sensorSelectionContainerLayout.addWidget(sensorSelectionLabel)
        sensorSelectionContainerLayout.addWidget(popUpWindow.sensorDropdown)

        # dropdown menu: last iteration or minimum loss selection
        popUpWindow.optimizationDropdown = QComboBox()
        popUpWindow.optimizationDropdown.addItems(
            [LAST_ITERATION_STRING, COLUMN_LABEL_MINIMUM_ERROR]
        )
        popUpWindow.optimizationDropdown.currentTextChanged.connect(
            lambda: self.onOptimizedSceneSelectorTextChanged(popUpWindow)
        )

        # output button
        outputBtn = QPushButton(text=OUTPUT_TO_JSON_STRING)
        outputBtn.clicked.connect(lambda: self.onOutputBtnPressed(popUpWindow))

        centralWidgetContainerLayout.addWidget(
            popUpWindow.optimizationDropdown
        )
        centralWidgetContainerLayout.addWidget(sensorSelectionContainer)
        centralWidgetContainerLayout.addWidget(outputBtn)
        popUpWindow.setCentralWidget(centralWidgetContainer)

        popUpWindow.show()

        lastIteration = len(popUpWindow.sceneParamsHist) - 1
        self.showOptimizedPlot(popUpWindow, lastIteration)

    def onOutputBtnPressed(self, popUpWindow: PopUpWindow):
        selectedIteration = len(popUpWindow.sceneParamsHist) - 1
        if (
            popUpWindow.optimizationDropdown.currentText()
            == COLUMN_LABEL_MINIMUM_ERROR
        ):
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
                    if len(v) > 1:
                        # special output: multi dimensional mi.Float as numpy array
                        outputNDimArrayFileName = (
                            outputFileDir
                            + f"/optimized_multi_dimensional_array_{k}.npy"
                        )
                        np.save(outputNDimArrayFileName, np.array(floatArray))
                    else:
                        # otherwise: fill the dictionary
                        outputDict[k] = floatArray
                else:
                    # default case: fill the dictionary
                    outputDict[k] = str(v)

            # output: filled dictionary
            json.dump(outputDict, outfile, indent=4)

            # output: resulting figure
            outputFileName = outputFileDir + "/figure.png"
            isVertical = False
            canvas = MplCanvas(isVertical)
            currentSensorIdx = int(popUpWindow.sensorDropdown.currentText())
            self.preparePlot(
                canvas,
                self.model.sensorToReferenceImageDict[
                    self.model.scene.sensors()[currentSensorIdx]
                ],
                popUpWindow.initImg,
                MaterialOptimizerModel.convertToBitmap(
                    MaterialOptimizerModel.render(
                        self.model.scene,
                        self.model.scene.sensors()[currentSensorIdx],
                        512,
                    )
                ),
                {self.model.lossFunction: popUpWindow.lossHist},
                selectedIteration,
                popUpWindow.lossHist[selectedIteration],
                isVertical,
            )
            import matplotlib.pyplot

            matplotlib.pyplot.savefig(outputFileName)

            # inform user
            absPath = str(Path(outputFileDir).resolve())
            self.view.showInfoMessageBox(
                f"The output can be found at: '{absPath}'"
            )

    def onOptimizedSceneSelectorTextChanged(self, popUpWindow: PopUpWindow):
        if (
            popUpWindow.optimizationDropdown.currentText()
            == COLUMN_LABEL_MINIMUM_ERROR
        ):
            self.showOptimizedPlot(
                popUpWindow,
                MaterialOptimizerModel.minIdxInDrList(popUpWindow.lossHist),
            )
        else:
            popUpWindow.showOptimizedPlot(len(popUpWindow.sceneParamsHist) - 1)

    def onSensorIdxChanged(self, popUpWindow: PopUpWindow):
        self.showOptimizedPlot(
            popUpWindow,
            MaterialOptimizerModel.minIdxInDrList(popUpWindow.lossHist),
            int(popUpWindow.sensorDropdown.currentText()),
        )

    def showOptimizedPlot(
        self, popUpWindow: PopUpWindow, iteration: int, sensorIdx: int = 0
    ):
        logging.info(
            f"Scene parameters in {iteration}:\n {popUpWindow.sceneParamsHist[iteration]}"
        )
        self.model.sceneParams.update(
            values=popUpWindow.sceneParamsHist[iteration]
        )
        sc = MplCanvas()
        self.preparePlot(
            sc,
            self.model.sensorToReferenceImageDict[
                self.model.scene.sensors()[sensorIdx]
            ],
            popUpWindow.initImg,
            MaterialOptimizerModel.convertToBitmap(
                MaterialOptimizerModel.render(
                    self.model.scene,
                    self.model.scene.sensors()[sensorIdx],
                    512,
                )
            ),
            {self.model.lossFunction: popUpWindow.lossHist},
            iteration,
            popUpWindow.lossHist[iteration],
        )
        sc.show()

    def preparePlot(
        self,
        canvas,
        refImage,
        initImg,
        finalImg,
        paramErrors,
        iterationNumber,
        loss,
        isVertical=True,
    ):
        if isVertical:
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
        else:
            for k, v in paramErrors.items():
                canvas.axes[0].plot(v, label=k)

            canvas.axes[0].set_xlabel(ITERATION_STRING)
            canvas.axes[0].set_ylabel(LOSS_STRING)
            canvas.axes[0].legend()
            canvas.axes[0].set_title(PARAMETER_ERROR_PLOT_STRING)

            canvas.axes[1].imshow(
                MaterialOptimizerModel.convertToBitmap(initImg)
            )
            canvas.axes[1].axis(OFF_STRING)
            canvas.axes[1].set_title(INITIAL_IMAGE_STRING)

            canvas.axes[2].imshow(
                MaterialOptimizerModel.convertToBitmap(finalImg)
            )
            canvas.axes[2].axis(OFF_STRING)
            canvas.axes[2].set_title(
                f"Optimized image: Iteration #{iterationNumber}, Loss: {loss:6f}"
            )

            canvas.axes[3].imshow(
                MaterialOptimizerModel.convertToBitmap(refImage)
            )
            canvas.axes[3].axis(OFF_STRING)
            canvas.axes[3].set_title(REFERENCE_IMAGE_STRING)
