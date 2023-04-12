import datetime
import json
import sys
import mitsuba as mi
import drjit as dr
import logging
from PyQt6.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
from src.constants import *
from src.material_optimizer_model import MaterialOptimizerModel
from src.material_optimizer_view import (
    MaterialOptimizerView,
    MplCanvas,
    PopUpWindow,
)
from PIL import Image, ImageChops, ImageOps
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
            self.view.tableContainer.setDisabled(True)
            self.view.bottomContainer.setDisabled(True)
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
            self.view.tableContainer.setDisabled(False)
            self.view.bottomContainer.setDisabled(False)
        except Exception as err:
            logging.error(str(err))
            self.view.showInfoMessageBox(
                f"Cannot load reference image/s. {str(err)}"
            )

    def updateSignals(self):
        self.view.table.cellChanged.connect(self.onCellChanged)
        self.view.table.verticalHeader().sectionDoubleClicked.connect(
            self.onVerticalHeaderSectionDoubleClicked
        )
        self.view.table.verticalHeader().sectionPressed.connect(
            self.onVerticalHeaderSectionPressed
        )

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
            self.view.setDisabled(True)
            self.optimizeMaterials()
            self.view.setDisabled(False)
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

        opts, sensorToInitImg = self.model.prepareOptimization(checkedRows)

        self.view.progressBar.setValue(100)
        self.view.progressBar.reset()

        # initiate the optimization loop
        (
            lossHist,
            sceneParamsHist,
            optLog,
            diffRenderHist,
        ) = self.model.optimizationLoop(
            opts,
            lambda x: self.view.progressBar.setValue(x),
            self.view.showDiffRender,
        )

        self.view.progressBar.setValue(self.view.progressBar.maximum())
        logging.info(f"Initial scene parameters:\n {sceneParamsHist[0]}")

        if len(lossHist) <= 1:
            self.view.showInfoMessageBox("No optimization was necessary")
        else:
            popUp = PopUpWindow(self.view)
            self.initOptimizedSceneSelector(
                popUp,
                sensorToInitImg,
                lossHist,
                sceneParamsHist,
                optLog,
                diffRenderHist,
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
                else:
                    self.askUserToResetSceneParamToInitValue(
                        currentSceneParams, key, initValue
                    )
            elif valueType is mi.TensorXf:
                self.askUserToResetSceneParamToInitValue(
                    currentSceneParams, key, initValue
                )

        return currentSceneParams

    def askUserToResetSceneParamToInitValue(
        self, currentSceneParams, key, initValue
    ):
        msg = "It seems this is not the first optimization. "
        msg += f"Should we reset the parameter value of {key} to "
        msg += "its initial value?"
        isYes = self.view.getUserDecision("Question", msg)
        if isYes:
            currentSceneParams[key] = initValue

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
        self.view.marginPercentageLine.editingFinished.connect(
            self.onMarginPercentageChanged
        )
        self.view.marginPenalty.currentTextChanged.connect(
            self.onMarginPenaltyChanged
        )
        self.view.table.verticalHeader().sectionDoubleClicked.connect(
            self.onVerticalHeaderSectionDoubleClicked
        )
        self.view.table.verticalHeader().sectionPressed.connect(
            self.onVerticalHeaderSectionPressed
        )

    def onVerticalHeaderSectionDoubleClicked(self, rowIndex):
        rowLabel = self.getRowLabelText()
        from PyQt6.QtGui import QColor

        if self.model.optimizationParams[rowLabel][COLUMN_LABEL_OPTIMIZE]:
            self.model.optimizationParams[rowLabel][
                COLUMN_LABEL_OPTIMIZE
            ] = False
            self.view.table.setStyleSheet(
                self.view.getQTableItemSelectedStyle(TOL_BLUE_COLOR)
            )
            for colIdx in range(self.view.table.columnCount()):
                self.view.table.item(rowIndex, colIdx).setBackground(
                    QColor(WHITE_COLOR)
                )
        else:
            self.model.optimizationParams[rowLabel][
                COLUMN_LABEL_OPTIMIZE
            ] = True
            self.view.table.setStyleSheet(
                self.view.getQTableItemSelectedStyle(TOL_GOLD_COLOR)
            )
            for colIdx in range(self.view.table.columnCount()):
                self.view.table.item(rowIndex, colIdx).setBackground(
                    QColor(TOL_GOLD_COLOR)
                )

    def onVerticalHeaderSectionPressed(self):
        rowLabel = self.getRowLabelText()

        if self.model.optimizationParams[rowLabel][COLUMN_LABEL_OPTIMIZE]:
            self.view.table.setStyleSheet(
                self.view.getQTableItemSelectedStyle(TOL_GOLD_COLOR)
            )
        else:
            self.view.table.setStyleSheet(
                self.view.getQTableItemSelectedStyle(TOL_BLUE_COLOR)
            )

    def onMarginPercentageChanged(self):
        try:
            self.model.setMarginPercentage(
                self.view.marginPercentageLine.text()
            )
        except Exception as err:
            self.view.marginPercentageLine.setText(INF_STR)
            self.view.showInfoMessageBox(str(err))

    def onMarginPenaltyChanged(self, text: str):
        self.model.setMarginPenalty(text)

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
        for param in list(self.model.optimizationParams.keys()):
            if self.model.optimizationParams[param][COLUMN_LABEL_OPTIMIZE]:
                result.append(param)
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

    def isLegalBetaValue(
        self, row: int, col: int, paramCol: str, newValue: float
    ):
        if paramCol == COLUMN_LABEL_BETA_1 or paramCol == COLUMN_LABEL_BETA_2:
            if newValue > MAX_BETA_VALUE:
                msg = "Optimization parameter is not changed. Please make"
                msg += " sure that the beta value is not larger than"
                msg += f" '{MAX_BETA_VALUE}'."
                self.view.showInfoMessageBox(msg)
                self.view.table.item(row, col).setText(str(MAX_BETA_VALUE))
                return False
            elif newValue < MIN_BETA_VALUE:
                msg = "Optimization parameter is not changed. Please make"
                msg += " sure that the beta value is not smaller than"
                msg += f" '{MIN_BETA_VALUE}'."
                self.view.showInfoMessageBox(msg)
                self.view.table.item(row, col).setText(str(MIN_BETA_VALUE))
                return False

        return True

    def onMinOrMaxClampValueOfVertexPositionChanged(
        self, row, col, paramRow, paramCol
    ):
        # assumption: we are sure that it is vertex pos' and min or max clamp
        try:
            newValue = self.model.stringToPoint3f(
                self.view.table.item(row, col).text()
            )
            return True, paramRow, paramCol, newValue
        except ValueError as err:
            if paramCol == COLUMN_LABEL_MIN_CLAMP_LABEL:
                self.view.table.item(row, col).setText(
                    MaterialOptimizerView.Point3fToCellString(
                        (-MAX_VERTEX_POSITION_VALUE)
                    )
                )
            else:
                self.view.table.item(row, col).setText(
                    MaterialOptimizerView.Point3fToCellString(
                        (MAX_VERTEX_POSITION_VALUE)
                    )
                )
            errMsg = "Optimization parameter is not changed. " + str(err)
            self.view.showInfoMessageBox(errMsg)
            return False, None, None, None

    def onOptimizationParamChanged(self, row, col):
        paramRow = self.getRowLabelText()
        paramCol = self.getColumnLabelText()

        # make sure the column refers to optimization parameter
        if paramCol not in self.model.optimizationParams[paramRow]:
            return False, None, None, None

        try:
            # special case: vertex_
            if VERTEX_POSITIONS_PATTERN.search(
                paramRow
            ) or VERTEX_NORMALS_PATTERN.search(paramRow):
                if (
                    paramCol == COLUMN_LABEL_MIN_CLAMP_LABEL
                    or paramCol == COLUMN_LABEL_MAX_CLAMP_LABEL
                ):
                    # we are sure that it is vertex pos. and min or max clamp val.
                    return self.onMinOrMaxClampValueOfVertexPositionChanged(
                        row, col, paramRow, paramCol
                    )

            newValue = float(self.view.table.item(row, col).text())
            if not self.isLegalLearningRate(row, col, paramCol, newValue):
                return False, None, None, None
            if not self.isLegalClampValue(
                col, row, paramCol, paramRow, newValue
            ):
                return False, None, None, None
            if not self.isLegalBetaValue(row, col, paramCol, newValue):
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
        self,
        popUpWindow: PopUpWindow,
        sensorToInitImg,
        lossHist,
        sceneParamsHist,
        optLog,
        diffRenderHist,
    ):

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
                for sensorIdx in range(len(self.model.scene.sensors()))
            ]
        )
        popUpWindow.sensorDropdown.currentTextChanged.connect(
            lambda: self.onSensorIdxChanged(
                popUpWindow, lossHist, sceneParamsHist, sensorToInitImg
            )
        )
        sensorSelectionContainerLayout.addWidget(sensorSelectionLabel)
        sensorSelectionContainerLayout.addWidget(popUpWindow.sensorDropdown)

        # dropdown menu: last iteration or minimum loss selection
        popUpWindow.optimizationDropdown = QComboBox()
        popUpWindow.optimizationDropdown.addItems(
            [LAST_ITERATION_STRING, COLUMN_LABEL_MINIMUM_ERROR]
        )
        popUpWindow.optimizationDropdown.currentTextChanged.connect(
            lambda: self.onOptimizedSceneSelectorTextChanged(
                popUpWindow, lossHist, sceneParamsHist, sensorToInitImg
            )
        )

        # output button
        outputBtn = QPushButton(text=OUTPUT_TO_JSON_STRING)
        outputBtn.clicked.connect(
            lambda: self.onOutputBtnPressed(
                popUpWindow,
                lossHist,
                sceneParamsHist,
                sensorToInitImg,
                optLog,
                diffRenderHist,
            )
        )

        centralWidgetContainerLayout.addWidget(
            popUpWindow.optimizationDropdown
        )
        centralWidgetContainerLayout.addWidget(sensorSelectionContainer)
        centralWidgetContainerLayout.addWidget(outputBtn)
        popUpWindow.setCentralWidget(centralWidgetContainer)

        popUpWindow.show()

        lastIteration = len(sceneParamsHist) - 1
        self.showOptimizedPlot(
            sceneParamsHist, sensorToInitImg, lossHist, lastIteration
        )

    def onOutputBtnPressed(
        self,
        popUpWindow: PopUpWindow,
        lossHist,
        sceneParamsHist,
        sensorToInitImg,
        optLog,
        diffRenderHist,
    ):
        selectedIteration = len(sceneParamsHist) - 1
        if (
            popUpWindow.optimizationDropdown.currentText()
            == COLUMN_LABEL_MINIMUM_ERROR
        ):
            selectedIteration = MaterialOptimizerModel.minIdxInDrList(lossHist)

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
            for k, v in sceneParamsHist[selectedIteration].items():
                if type(v) is mi.TensorXf:
                    # special output: volume
                    if ALBEDO_DATA_PATTERN.search(k):
                        self.outputVolume(outputFileDir, k, v)
                    elif GRID_VOLUME_TO_OPTIMIZER_PATTERN.search(k):
                        self.outputVolume(outputFileDir, k, v)

                        # apply marching cubes
                        from skimage import measure

                        valNpy = np.array(v)[:, :, :, 0]
                        verts, faces, normals, values = measure.marching_cubes(
                            valNpy,
                            0.005,
                            allow_degenerate=False,
                        )

                        verts, faces = self.convert_obj_to_br(
                            verts, faces, valNpy.shape[0]
                        )
                        outputObjName = (
                            outputFileDir + f"//optimized_volume_{k}.obj"
                        )
                        self.marching_cubes_to_obj(
                            (verts, faces, normals, values), outputObjName
                        )
                    else:
                        # special output: bitmap texture
                        outputTextureFileName = (
                            outputFileDir + f"/optimized_texture_{k}.png"
                        )
                        mi.util.write_bitmap(outputTextureFileName, v)
                elif type(v) is mi.Float:
                    floatArray = [f for f in v]
                    if VERTEX_POSITIONS_PATTERN.search(k):
                        if any(
                            k in checkedRow
                            for checkedRow in self.getCheckedRows()
                        ):
                            self.outputPlyMesh(outputFileDir, k)
                    elif len(v) > 1:
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

            # prepare figure content and output each element
            sensorToOptimizedImage = {}
            self.prepareFigureAndOutputEachElem(
                lossHist,
                sensorToInitImg,
                outputFileDir,
                sensorToOptimizedImage,
                selectedIteration,
                diffRenderHist,
            )

            # output: resulting figure
            figuresDir = outputFileDir + f"/figures_it_{selectedIteration}"
            Path(figuresDir).mkdir(parents=True, exist_ok=True)
            for sensorIdx in range(len(self.model.scene.sensors())):
                self.outputFigure(
                    selectedIteration,
                    figuresDir,
                    sensorIdx,
                    lossHist,
                    sensorToInitImg,
                    sensorToOptimizedImage,
                )

            # output: optimization log file
            optLogFileName = outputFileDir + "/optimization.log"
            with open(optLogFileName, "w") as f:
                f.write(optLog)

            # inform user
            absPath = str(Path(outputFileDir).resolve())
            self.view.showInfoMessageBox(
                f"The output can be found at: '{absPath}'"
            )

    def outputPlyMesh(self, outputFileDir, k):
        if ".vertex_positions" not in k:
            return

        parentStr = k.replace(".vertex_positions", "")
        mesh = mi.Mesh(
            "optimized_mesh",
            vertex_count=self.model.sceneParams[parentStr + ".vertex_count"],
            face_count=self.model.sceneParams[parentStr + ".face_count"],
            has_vertex_normals=True,
            has_vertex_texcoords=False,
        )
        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = dr.ravel(self.model.sceneParams[k])
        mesh_params["vertex_normals"] = dr.ravel(
            self.model.sceneParams[parentStr + ".vertex_normals"]
        )
        mesh_params["faces"] = dr.ravel(
            self.model.sceneParams[parentStr + ".faces"]
        )
        print(mesh_params.update())
        outputMeshName = outputFileDir + f"//optimized_mesh_{k}.ply"
        mesh.write_ply(outputMeshName)

    def prepareFigureAndOutputEachElem(
        self,
        lossHist,
        sensorToInitImg,
        outputFileDir,
        sensorToOptimizedImage,
        selectedIteration,
        diffRenderHist,
    ):
        refImgsDir = outputFileDir + "/ref_imgs"
        Path(refImgsDir).mkdir(parents=True, exist_ok=True)
        initImgsDir = outputFileDir + "/init_imgs"
        Path(initImgsDir).mkdir(parents=True, exist_ok=True)
        optImgsDir = outputFileDir + f"/opt_imgs_it_{selectedIteration}"
        Path(optImgsDir).mkdir(parents=True, exist_ok=True)
        absErrImgs = outputFileDir + f"/abs_err_imgs_it_{selectedIteration}"
        Path(absErrImgs).mkdir(parents=True, exist_ok=True)
        diffRenderImgs = outputFileDir + "/diff_render_history"
        Path(diffRenderImgs).mkdir(parents=True, exist_ok=True)

        for sensorIdx in range(len(self.model.scene.sensors())):
            currentSensor = self.model.scene.sensors()[sensorIdx]

            # output: reference image
            refImgName = refImgsDir + f"/ref_img_s{sensorIdx}.png"
            mi.util.write_bitmap(
                refImgName,
                self.model.sensorToReferenceImageDict[currentSensor],
            )

            # output: init image
            initImgName = initImgsDir + f"/init_img_s{sensorIdx}.png"
            mi.util.write_bitmap(initImgName, sensorToInitImg[currentSensor])

            sensorToOptimizedImage[
                sensorIdx
            ] = MaterialOptimizerModel.convertToBitmap(
                MaterialOptimizerModel.render(
                    self.model.scene,
                    currentSensor,
                    512,
                )
            )
            # output: opt image
            optImageName = optImgsDir + f"/opt_img_s{sensorIdx}.png"
            mi.util.write_bitmap(
                optImageName,
                sensorToOptimizedImage[sensorIdx],
            )

            # output: abs error image
            refBmp = MaterialOptimizerModel.convertToBitmap(
                self.model.sensorToReferenceImageDict[currentSensor]
            )
            absErrImg = self.getAbsoluteErrImage(
                sensorToOptimizedImage[sensorIdx], refBmp
            )
            absErrImageName = absErrImgs + f"/abs_err_img_s{sensorIdx}.png"
            plt.imsave(absErrImageName, absErrImg, cmap="inferno")

            # output: diff render history
            diffRenderSensor = diffRenderImgs + f"/sensor_{sensorIdx}"
            Path(diffRenderSensor).mkdir(parents=True, exist_ok=True)
            for it, diffRender in enumerate(diffRenderHist[sensorIdx]):
                imgName = diffRenderSensor + f"/{it}.png"
                mi.util.write_bitmap(
                    imgName,
                    diffRender,
                )

        # output: loss history
        np.save((outputFileDir + "/loss_histroy.npy"), np.array(lossHist))

    def outputVolume(self, outputFileDir, k, v):
        outputVolumeFileName = outputFileDir + f"//optimized_volume_{k}.vol"
        mi.VolumeGrid(v).write(outputVolumeFileName)

    def outputFigure(
        self,
        selectedIteration,
        outputFileDir,
        sensorIdx,
        lossHist,
        sensorToInitImg,
        sensorToOptimizedImage,
    ):
        outputFileName = outputFileDir + f"/figure-sensor{sensorIdx}.png"
        canvas = MplCanvas()
        currentSensor = self.model.scene.sensors()[sensorIdx]
        self.preparePlot(
            canvas,
            self.model.sensorToReferenceImageDict[currentSensor],
            sensorToInitImg[currentSensor],
            sensorToOptimizedImage[sensorIdx],
            {self.model.lossFunction: lossHist},
            selectedIteration,
            lossHist[selectedIteration],
        )
        plt.savefig(outputFileName)

    def onOptimizedSceneSelectorTextChanged(
        self,
        popUpWindow: PopUpWindow,
        lossHist,
        sceneParamsHist,
        sensorToInitImg,
    ):
        if (
            popUpWindow.optimizationDropdown.currentText()
            == COLUMN_LABEL_MINIMUM_ERROR
        ):
            self.showOptimizedPlot(
                sceneParamsHist,
                sensorToInitImg,
                lossHist,
                MaterialOptimizerModel.minIdxInDrList(lossHist),
            )
        else:
            self.showOptimizedPlot(
                sceneParamsHist,
                sensorToInitImg,
                lossHist,
                len(sceneParamsHist) - 1,
            )

    def onSensorIdxChanged(
        self, popUpWindow, lossHist, sceneParamsHist, sensorToInitImg
    ):
        self.showOptimizedPlot(
            sceneParamsHist,
            sensorToInitImg,
            lossHist,
            MaterialOptimizerModel.minIdxInDrList(lossHist),
            int(popUpWindow.sensorDropdown.currentText()),
        )

    def showOptimizedPlot(
        self,
        sceneParamsHist,
        sensorToInitImg,
        lossHist,
        iteration: int,
        sensorIdx: int = 0,
    ):
        logging.info(
            f"Scene parameters in {iteration}:\n {sceneParamsHist[iteration]}"
        )
        self.model.sceneParams.update(values=sceneParamsHist[iteration])
        sc = MplCanvas()
        currentSensor = self.model.scene.sensors()[sensorIdx]
        self.preparePlot(
            sc,
            self.model.sensorToReferenceImageDict[currentSensor],
            sensorToInitImg[currentSensor],
            MaterialOptimizerModel.convertToBitmap(
                MaterialOptimizerModel.render(
                    self.model.scene,
                    currentSensor,
                    512,
                )
            ),
            {self.model.lossFunction: lossHist},
            iteration,
            lossHist[iteration],
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
    ):
        for k, v in paramErrors.items():
            canvas.axes[0].plot(v, label=k)

        canvas.axes[0].set_xlabel(ITERATION_STRING)
        canvas.axes[0].set_ylabel(LOSS_STRING)
        canvas.axes[0].legend()
        canvas.axes[0].set_title(PARAMETER_ERROR_PLOT_STRING)

        canvas.axes[1].imshow(MaterialOptimizerModel.convertToBitmap(initImg))
        canvas.axes[1].axis(OFF_STRING)
        canvas.axes[1].set_title(INITIAL_IMAGE_STRING)

        canvas.axes[2].imshow(finalImg)
        canvas.axes[2].axis(OFF_STRING)
        canvas.axes[2].set_title(
            f"Optimized image: Iteration #{iterationNumber}, Loss: {loss:6f}"
        )

        refBmp = MaterialOptimizerModel.convertToBitmap(refImage)
        canvas.axes[3].imshow(refBmp)
        canvas.axes[3].axis(OFF_STRING)
        canvas.axes[3].set_title(REFERENCE_IMAGE_STRING)

        diff_np = self.getAbsoluteErrImage(finalImg, refBmp)
        im4 = canvas.axes[4].imshow(diff_np, cmap="inferno")
        divider = make_axes_locatable(canvas.axes[4])
        cax4 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im4, cax=cax4)
        canvas.axes[4].axis(OFF_STRING)
        canvas.axes[4].set_title(r"Absolute Error: |$y_{ref} - y_{opt}$|")

        plt.tight_layout()

    @staticmethod
    def getAbsoluteErrImage(finalImg, refBmp):
        refPilImg = Image.fromarray(np.uint8(refBmp))
        finalPilImg = Image.fromarray(np.uint8(finalImg))
        diff_np = MaterialOptimizerController.normalized_absolute_error(
            refPilImg, finalPilImg
        )
        return diff_np

    @staticmethod
    def normalized_absolute_error(ref: Image, opt: Image):
        diff = ImageChops.difference(ref, opt)
        diff = ImageOps.grayscale(diff)
        diff_np = np.array(diff, dtype="float64")
        diff_np *= 1.0 / diff_np.max()
        return diff_np

    @staticmethod
    def convert_obj_to_br(verts, faces, voxel_size):
        """
        Note from Hasbay: Code taken from:
        https://programtalk.com/vs4/python/brainglobe/brainreg-segment/brainreg_segment/regions/IO.py/
        """
        if voxel_size != 1:
            verts = verts * voxel_size

        faces = faces + 1
        return verts, faces

    @staticmethod
    def marching_cubes_to_obj(marching_cubes_out, output_file):
        """
        Note from Hasbay: Code taken from:
        https://programtalk.com/vs4/python/brainglobe/brainreg-segment/brainreg_segment/regions/IO.py/
        Saves the output of skimage.measure.marching_cubes as an .obj file
        :param marching_cubes_out: tuple
        :param output_file: str
        """

        verts, faces, normals, _ = marching_cubes_out
        with open(output_file, "w") as f:
            for item in verts:
                f.write(f"v {item[0]} {item[1]} {item[2]}\n")
            for item in normals:
                f.write(f"vn {item[0]} {item[1]} {item[2]}\n")
            for item in faces:
                f.write(
                    f"f {item[0]}//{item[0]} {item[1]}//{item[1]} "
                    f"{item[2]}//{item[2]}\n"
                )
            f.close()
