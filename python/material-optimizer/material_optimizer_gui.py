"""
Author: Can Hasbay
"""

from PyQt6.QtWidgets import *
from PyQt6.QtGui import QAction
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import sys
import mitsuba as mi
import drjit as dr
import logging
from PyQt6 import QtGui, QtWidgets, QtCore
import ctypes
from constants import *

mi.set_variant("cuda_ad_rgb")


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.refImage = None
        self.sceneRes = (256, 256)
        self.loadMitsubaScene()
        self.setSceneParams(self.scene)
        self.setInitialSceneParams(self.sceneParams)
        self.setDefaultOptimizationParams(self.initialSceneParams)

    def setScene(self, fileName: str, sceneRes: tuple, integratorType: str):
        self.scene = mi.load_file(
            fileName,
            resx=sceneRes[0],
            resy=sceneRes[1],
            width=sceneRes[0],
            height=sceneRes[1],
            resolution=sceneRes,
            integrator=integratorType,
        )

    def loadMitsubaScene(self, fileName=None):
        if fileName is None:
            fileName = SCENES_DIR_PATH + "cbox.xml"

        self.integratorType = self.findPossibleIntegratorType(fileName)
        self.setScene(fileName, self.sceneRes, self.integratorType)
        self.fileName = fileName

    def findPossibleIntegratorType(self, fileName) -> str:
        tmpScene = mi.load_file(fileName)
        tmpParams = mi.traverse(tmpScene)
        integratorType = "prb"
        if self.anyParamsProducesDiscontinuities(tmpParams):
            integratorType = "prb_reparam"
        return integratorType

    def anyParamsProducesDiscontinuities(self, tmpParams):
        return any(
            pattern.search(k)
            for k in tmpParams.properties
            for pattern in PATTERNS_INTRODUCE_DISCONTINUITIES
        )

    def resetReferenceImage(self):
        if self.refImage is not None:
            self.refImage = None

    def getDefaultOptimizationParams(self, params):
        return {
            param: {
                "Learning Rate": 0.3,
                "Minimum Error": 0.01,
                "Optimize": False,
            }
            for param in params
        }

    def setDefaultOptimizationParams(self, initialParams):
        self.optimizationParams = self.getDefaultOptimizationParams(
            initialParams
        )

    def setInitialSceneParams(self, params):
        self.initialSceneParams = dict(
            self.createSubsetSceneParams(params, SUPPORTED_BSDF_PATTERNS)
        )

    def setSceneParams(self, scene: mi.Scene):
        self.sceneParams = mi.traverse(scene)

    def getModifiedParams(self):
        result = {}
        for refKey, refValue in self.initialSceneParams.items():
            if self.sceneParams[refKey] != refValue:
                result[refKey] = self.sceneParams[refKey]
        return result

    def createSubsetSceneParams(
        self, params: mi.SceneParameters, patterns: list
    ) -> dict:
        result = {}
        for k, v in params.items():
            for pattern in patterns:
                if pattern.search(k):
                    self.copyMitsubaTypeIfSupported(result, k, v)
        return result

    def copyMitsubaTypeIfSupported(self, result, k, v):
        vType = type(v)
        if vType is mi.Color3f:
            result[k] = mi.Color3f(v)
        elif vType is mi.Float:
            if len(v) == 1:
                result[k] = mi.Float(v[0])

    def updateParamErrors(
        self, params, initialParams, modifiedParams, paramErrors
    ):
        for key in modifiedParams:
            err = dr.sum(dr.sqr(initialParams[key] - params[key]))[0]
            paramErrors[key].append(err)
            logging.info(f"\tkey= {key} error= {paramErrors[key][-1]:6f}")

    def updateAfterStep(self, opts, params):
        for opt in opts:
            # Optimizer: take a gradient descent step
            opt.step()
            for key in opt.keys():
                self.ensureLegalParamValues(opt, key)
            # Update the scene state to the new optimized values
            params.update(opt)

    def ensureLegalParamValues(self, opt, key):
        # Post-process the optimized parameters to ensure legal values
        if ETA_PATTERN.search(key):
            opt[key] = dr.clamp(opt[key], 0.0, 3.0)
        elif DIFF_TRANS_PATTERN.search(key):
            opt[key] = dr.clamp(opt[key], 0.0, 2.0)
        else:
            opt[key] = dr.clamp(opt[key], 0.0, 1.0)

    def mse(self, image, refImage):
        return dr.mean(dr.sqr(refImage - image))

    def getMinParamErrors(self, optimizationParams: dict):
        return {
            sceneParam: optimizationParam["Minimum Error"]
            for sceneParam, optimizationParam in optimizationParams.items()
        }

    def getInitParamErrors(self, modifiedParams):
        return {
            k: [
                dr.sum(
                    dr.sqr(self.initialSceneParams[k] - self.sceneParams[k])
                )[0]
            ]
            for k in modifiedParams
        }

    def initOptimizersWithCustomValues(self, customParams: list) -> list:
        opt = mi.ad.Adam(
            lr=0.2, params={k: self.sceneParams[k] for k in customParams}
        )
        opt.set_learning_rate(
            {
                k: self.optimizationParams[k]["Learning Rate"]
                for k in customParams
            }
        )
        return [opt]

    def render(self, scene: mi.Scene, spp: int, params=None, seed=0):
        return mi.render(scene, params, seed=seed, spp=spp)

    def updateSceneParam(self, key, value):
        if self.sceneParams[key] == value:
            return

        self.sceneParams[key] = value
        self.sceneParams.update()

    def updateSceneParameters(self, currentSceneParams):
        for key in currentSceneParams:
            self.sceneParams[key] = currentSceneParams[key]
        self.sceneParams.update()

    def setReferenceImageToInitiallyLoadedScene(self):
        # remember modified params
        modifiedParams = self.copyModifiedParams()
        self.setSceneParamsToInitialParams()
        self.refImage = mi.render(self.scene, spp=512)
        self.updateSceneParameters(modifiedParams)

    def copyModifiedParams(self) -> dict:
        result = {}
        for k, v in self.getModifiedParams().items():
            self.copyMitsubaTypeIfSupported(result, k, v)
        return result

    def setSceneParamsToInitialParams(self):
        for key in self.getModifiedParams():
            self.sceneParams[key] = self.initialSceneParams[key]
        self.sceneParams.update()

    def stringToColor3f(self, newValue) -> mi.Color3f:
        result = newValue.split(",")
        errMsg = (
            "Invalid RGB input. Please use the following format: 1.0, 1.0, 1.0"
        )
        if len(result) != 3:
            raise ValueError(errMsg)
        try:
            red = float(result[0])
            green = float(result[1])
            blue = float(result[2])
            return mi.Color3f(red, green, blue)
        except:
            raise ValueError(errMsg)

    def updateOptimizationParam(self, paramCol, paramRow, newValue):
        if self.optimizationParams[paramRow][paramCol] == newValue:
            return
        self.optimizationParams[paramRow][paramCol] = newValue

    def readImage(self, fileName) -> mi.TensorXf:
        result: mi.Bitmap = mi.Bitmap(fileName).convert(
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False
        )
        resultSize = result.size()
        if resultSize != self.sceneRes:
            result = self.adjustImageSize(result, resultSize)
        result = mi.TensorXf(result)
        result = dr.clamp(result, 0.0, 1.0)
        return result

    def adjustImageSize(self, result, resultSize) -> mi.Bitmap:
        aspectRatio = resultSize[0] / resultSize[1]
        newSize = (int(256 * aspectRatio), 256)
        self.setScene(self.fileName, newSize, self.integratorType)
        self.setSceneParams(self.scene)
        result = result.resample(newSize)
        return result

    def updateSceneParamsWithOptimizers(self, opts):
        for opt in opts:
            self.sceneParams.update(opt)

    @staticmethod
    def is_float(element: any) -> bool:
        """Taken from https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python/20929881#20929881"""
        # If you expect None to be passed:
        if element is None:
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(element: any) -> bool:
        if element is None:
            return False
        try:
            int(element)
            return True
        except ValueError:
            return False

    def setMinErrOnCustomImage(self, value: str):
        if not MaterialOptimizerModel.is_float(value):
            raise ValueError("Please provide a valid float value (e.g. 0.001)")
        self.minErrOnCustomImage = float(value)

    def setSamplesPerPixelOnCustomImage(self, value: str):
        if not MaterialOptimizerModel.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 4)")
        self.samplesPerPixelOnCustomImage = int(value)

    @staticmethod
    def minIdxInDrList(lis: list):
        minValue = dr.min(lis)
        return lis.index(minValue)


class MaterialOptimizerView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(QtGui.QIcon(IMAGES_DIR_PATH + "sloth.png"))
        self.show()

    def initUI(self):
        self.initWindowProperties()
        self.statusBar()
        self.initMenu()
        # self.colorPickerWindow = None

    def initWindowProperties(self):
        self.setGeometry(1024, 512, 1024, 512)
        self.setWindowTitle("Material Optimizer")

    def initMenu(self):
        # Menu Bar: Import File
        self.importFile = QAction("Import", self)
        self.importFile.setShortcut("Ctrl+I")
        self.importFile.setStatusTip("Import Mitsuba 3 Scene File")

        # Menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(self.importFile)

    def showFileDialog(self, filterStr: str):
        homeDir = str(Path.home())
        return QFileDialog.getOpenFileName(
            self, "Import File", homeDir, filterStr
        )[0]

    def showInfoMessageBox(self, text):
        msgBox = QMessageBox()
        msgBox.setText(text)
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
        msgBox.exec()

    def initCentralWidget(self, sceneParams: dict):
        centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout(centralWidget)

        self.initTopWidget(centralWidget)
        self.initTableContainer(sceneParams, centralWidget)
        self.initBottomContainer(centralWidget)

        self.centralLayout.addWidget(self.topWidget)
        self.centralLayout.addWidget(self.tableContainer)
        self.centralLayout.addWidget(self.bottomContainer)
        self.setCentralWidget(centralWidget)

    def initTableContainer(self, sceneParams, centralWidget):
        self.tableContainer = QWidget(centralWidget)
        self.tableContainerLayout = QVBoxLayout(self.tableContainer)
        self.table = self.initTable(sceneParams)
        self.tableContainerLayout.addWidget(self.table)

    def initTopWidget(self, centralWidget):
        self.defaultRefImgBtn = QRadioButton(
            "Use original scene as the reference image"
        )
        self.defaultRefImgBtn.setChecked(True)
        self.customRefImgBtn = QRadioButton("Use custom reference image")
        self.topWidget = QWidget(centralWidget)
        self.radioBtnLayout = QHBoxLayout(self.topWidget)
        self.radioBtnLayout.addWidget(self.defaultRefImgBtn)
        self.radioBtnLayout.addWidget(self.customRefImgBtn)

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
        minErrLabel = QLabel(text="Minimum Error")
        self.minErrLine = QLineEdit()
        self.minErrLine.setText(str(DEFAULT_MIN_ERR_ON_CUSTOM_IMG))
        self.minErrContainerLayout.addWidget(minErrLabel)
        self.minErrContainerLayout.addWidget(self.minErrLine)

        # samples per pixel dropdown
        self.sppContainer = QWidget(self.configContainer)
        self.sppContainerLayout = QHBoxLayout(self.sppContainer)
        samplesPerPixelLabel = QLabel(
            text="Samples per pixel during optimization"
        )
        self.samplesPerPixelBox = QComboBox()
        self.samplesPerPixelBox.addItems(SUPPORTED_SPP_VALUES)
        self.sppContainerLayout.addWidget(samplesPerPixelLabel)
        self.sppContainerLayout.addWidget(self.samplesPerPixelBox)

        self.configContainerLayout.addWidget(self.minErrContainer)
        self.configContainerLayout.addWidget(self.sppContainer)
        self.configContainer.hide()

    def initProgessContainer(self, centralWidget):
        self.progressContainer = QWidget(self.bottomContainer)
        self.progressContainerLayout = QHBoxLayout(self.progressContainer)
        self.progressBar = QProgressBar(self.bottomContainer)
        self.optimizeButton = QPushButton("Start Optimization", centralWidget)
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
                if label == "Value":
                    valueType = type(value)
                    if valueType is mi.Color3f:
                        itemContent = self.Color3fToCellString(value)
                        item = QTableWidgetItem(itemContent)
                    elif valueType is mi.Float:
                        if len(value) == 1:
                            item = QTableWidgetItem(str(value[0]))
                        else:
                            item = QTableWidgetItem("Not implemented yet")
                    else:
                        item = QTableWidgetItem("Not implemented yet")
                elif label == "Optimize":
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
        self.setWindowIcon(QtGui.QIcon(IMAGES_DIR_PATH + "sloth.png"))
        parent.setDisabled(True)
        self.setDisabled(False)

    def initOptimizedSceneSelector(
        self,
        model: MaterialOptimizerModel,
        initImg,
        lossHist: list,
        sceneParamsHist: list,
    ):
        comboBox = QComboBox()
        comboBox.addItems(["Last Iteration", "Min Error"])
        self.setCentralWidget(comboBox)
        comboBox.currentTextChanged.connect(
            self.onOptimizedSceneSelectorTextChanged
        )

        self.model = model
        self.initImg = initImg
        self.lossHist = lossHist
        self.sceneParamsHist = sceneParamsHist

        self.show()
        lastIteration = len(self.sceneParamsHist) - 1
        self.showOptimizedPlot(lastIteration)

    def onOptimizedSceneSelectorTextChanged(self, text: str):
        if text == "Min Error":
            self.showOptimizedPlot(
                MaterialOptimizerModel.minIdxInDrList(self.lossHist)
            )
        else:
            self.showOptimizedPlot(len(self.sceneParamsHist) - 1)

    def showOptimizedPlot(self, iteration: int):
        self.model.sceneParams.update(values=self.sceneParamsHist[iteration])
        sc = MplCanvas()
        lossHistKey = "MSE(Current-Image, Reference-Image)"
        lossHistDict = {lossHistKey: self.lossHist}
        sc.plotOptimizationResults(
            self.model.refImage,
            self.initImg,
            mi.util.convert_to_bitmap(
                self.model.render(self.model.scene, 512)
            ),
            lossHistDict,
            iteration,
            self.lossHist[iteration][0],
        )

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.parent().setDisabled(False)
        return super().closeEvent(a0)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        super(MplCanvas, self).__init__(fig)

    def plotOptimizationResults(
        self,
        refImage,
        initImg,
        finalImg,
        paramErrors,
        iterationNumber: int,
        loss: float,
    ):

        for k, v in paramErrors.items():
            self.axes[0][0].plot(v, label=k)

        self.axes[0][0].set_xlabel("iteration")
        self.axes[0][0].set_ylabel("Loss")
        self.axes[0][0].legend()
        self.axes[0][0].set_title("Parameter error plot")

        self.axes[0][1].imshow(mi.util.convert_to_bitmap(initImg))
        self.axes[0][1].axis("off")
        self.axes[0][1].set_title("Initial Image")

        self.axes[1][0].imshow(mi.util.convert_to_bitmap(finalImg))
        self.axes[1][0].axis("off")
        self.axes[1][0].set_title(
            f"Optimized image: Iteration #{iterationNumber}, Loss: {loss:6f}"
        )

        self.axes[1][1].imshow(mi.util.convert_to_bitmap(refImage))
        self.axes[1][1].axis("off")
        self.axes[1][1].set_title("Reference Image")
        plt.savefig(IMAGES_DIR_PATH + "material-optimizer-result-figure.png")

        self.show()


class MaterialOptimizerController:
    """Material Optimizer's controller class."""

    def __init__(
        self, model: MaterialOptimizerModel, view: MaterialOptimizerView
    ):
        self.model = model
        self.view = view
        tableValues = self.combineTableValues(
            self.model.initialSceneParams, self.model.optimizationParams
        )
        self.view.initCentralWidget(tableValues)
        self.hideTableColumn("Optimize", True)
        self.connectSignals()

    def loadMitsubaScene(self):
        try:
            fileName = self.view.showFileDialog("Xml File (*.xml)")
            self.model.loadMitsubaScene(fileName)
            self.model.resetReferenceImage()
            self.view.defaultRefImgBtn.setChecked(True)
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
            msg += f" Mitsuba Error: {err=}"
            self.view.showInfoMessageBox(msg)

    def updateTable(self, params, optimizationParams):
        newTable = self.view.initTable(
            self.combineTableValues(params, optimizationParams)
        )
        self.view.replaceTable(newTable)
        self.hideTableColumn("Optimize", True)
        self.updateSignals()

    def loadReferenceImage(self):
        try:
            refImgFileName = self.view.showFileDialog("Images (*.png *.jpg)")
            readImg = self.model.readImage(refImgFileName)
            self.model.refImage = readImg
            msg = "Selected image size is " + str(readImg.shape)
            msg += "Resampling the reference image according to the loaded"
            msg += " scene resolution, if necessary."
        except Exception as err:
            logging.error(str(err))
            errMsg = "Cannot load the reference image. Switching back to the"
            errMsg += " default reference image."
            self.view.defaultRefImgBtn.setChecked(True)
            self.view.showInfoMessageBox(errMsg)

    def updateSignals(self):
        self.view.table.cellChanged.connect(self.onCellChanged)

    def combineTableValues(self, params, optimizationParams):
        result = {}
        for key in params:
            result[key] = {"Value": params[key]}
            result[key].update(optimizationParams[key])
        return result

    def optimizeMaterials(self):
        if self.view.defaultRefImgBtn.isChecked():
            self.optimizeMaterialsWithDefaultReferenceImage()
        else:
            self.optimizeMaterialsWithCustomReferenceImage()

    def optimizeMaterialsWithDefaultReferenceImage(self):
        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()
            currentSceneParams = self.getCurrentSceneParams()
            self.model.updateSceneParameters(currentSceneParams)

        modifiedParams = self.model.getModifiedParams()
        if len(modifiedParams) <= 0:
            self.view.showInfoMessageBox(
                "Please modify a scene parameter before optimization."
            )
            return

        self.view.optimizeButton.setDisabled(True)
        if self.model.refImage is None:
            self.view.progressBar.setValue(25)
            self.model.setReferenceImageToInitiallyLoadedScene()
            self.view.progressBar.setValue(self.view.progressBar.maximum())
            self.view.progressBar.reset()

        self.view.progressBar.setValue(50)
        opts = self.model.initOptimizersWithCustomValues(list(modifiedParams))
        self.model.updateSceneParamsWithOptimizers(opts)
        initImg = self.model.render(self.model.scene, spp=256)
        paramErrors = self.model.getInitParamErrors(modifiedParams)
        minErrors = self.model.getMinParamErrors(self.model.optimizationParams)
        iterationCount = 200
        sceneParamsHist = []
        lossHist = []
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(iterationCount):
            self.view.progressBar.setValue(int(it / iterationCount * 100))

            # check all optimization parameters and if defined threshold is
            # achieved stop optimization for that parameter (i.e. pop optimization param)
            for opt in opts:
                for key in list(opt.keys()):
                    if key in opt and paramErrors[key][-1] < minErrors[key]:
                        opt.variables.pop(key)
                        logging.info(f"Key {key} is optimized")

            # stop optimization if all optimization variables are empty
            # (i.e. if all optimization params reached a defined threshold)
            if all(map(lambda opt: not opt.variables, opts)):
                break

            # Perform a (noisy) differentiable rendering of the scene
            image = self.model.render(
                self.model.scene, spp=4, params=self.model.sceneParams, seed=it
            )

            # Evaluate the objective function from the current rendered image
            loss = self.model.mse(image, self.model.refImage)
            lossHist.append(loss)
            sceneParamsHist.append(
                self.model.createSubsetSceneParams(
                    self.model.sceneParams, SUPPORTED_BSDF_PATTERNS
                )
            )

            # Backpropagate through the rendering process
            dr.backward(loss)

            self.model.updateAfterStep(opts, self.model.sceneParams)

            logging.info(f"Iteration {it:02d}")
            self.model.updateParamErrors(
                self.model.sceneParams,
                self.model.initialSceneParams,
                modifiedParams,
                paramErrors,
            )

        self.view.progressBar.setValue(self.view.progressBar.maximum())

        if it <= 0:
            self.view.showInfoMessageBox("No optimization was necessary")
        else:
            popUp = PopUpWindow(self.view)
            popUp.initOptimizedSceneSelector(
                self.model, initImg, lossHist, sceneParamsHist
            )

        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText("Restart Optimization")

    def optimizeMaterialsWithCustomReferenceImage(self):
        checkedRows = self.getCheckedRows()
        if len(checkedRows) <= 0:
            msg = "Please check at least one scene parameter for the optimization."
            self.view.showInfoMessageBox(msg)
            return

        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()
            currentSceneParams = self.getCurrentSceneParams()
            self.model.updateSceneParameters(currentSceneParams)

        self.view.optimizeButton.setDisabled(True)
        self.view.progressBar.setValue(50)
        opts = self.model.initOptimizersWithCustomValues(checkedRows)
        self.model.updateSceneParamsWithOptimizers(opts)
        initImg = self.model.render(self.model.scene, spp=256)
        iterationCount = 100
        sceneParamsHist = []
        lossHist = []
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(iterationCount):
            self.view.progressBar.setValue(int(it / iterationCount * 100))

            # Perform a (noisy) differentiable rendering of the scene
            image = self.model.render(
                self.model.scene,
                spp=self.model.samplesPerPixelOnCustomImage,
                params=self.model.sceneParams,
                seed=it,
            )
            image = dr.clamp(image, 0.0, 1.0)

            # Evaluate the objective function from the current rendered image
            loss = self.model.mse(image, self.model.refImage)
            lossHist.append(loss)
            sceneParamsHist.append(
                self.model.createSubsetSceneParams(
                    self.model.sceneParams, SUPPORTED_BSDF_PATTERNS
                )
            )
            logging.info(f"Iteration {it:02d}")
            logging.info(f"\tcurrent loss= {loss[0]:6f}")

            if loss[0] < self.model.minErrOnCustomImage:
                break

            # Backpropagate through the rendering process
            dr.backward(loss)

            self.model.updateAfterStep(opts, self.model.sceneParams)

        self.view.progressBar.setValue(self.view.progressBar.maximum())

        if it <= 0:
            self.view.showInfoMessageBox("No optimization was necessary")
        else:
            popUp = PopUpWindow(self.view)
            popUp.initOptimizedSceneSelector(
                self.model, initImg, lossHist, sceneParamsHist
            )

        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText("Restart Optimization")

    def getCurrentSceneParams(self):
        currentSceneParams = {}
        for row in range(self.view.table.rowCount()):
            key = self.view.table.verticalHeaderItem(row).text()
            initValue = self.model.initialSceneParams[key]
            newValue = self.view.table.item(row, 0).text()
            if type(initValue) is mi.Color3f:
                currentSceneParams[key] = self.model.stringToColor3f(newValue)
            elif type(initValue) is mi.Float:
                currentSceneParams[key] = mi.Float(float(newValue))

        return currentSceneParams

    def connectSignals(self):
        self.view.importFile.triggered.connect(self.loadMitsubaScene)
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)
        self.view.table.cellChanged.connect(self.onCellChanged)
        self.view.customRefImgBtn.toggled.connect(
            self.onCustomRefImgBtnChecked
        )
        self.view.defaultRefImgBtn.toggled.connect(
            self.onDefaultRefImgBtnChecked
        )
        self.view.minErrLine.editingFinished.connect(self.onMinErrLineChanged)
        self.view.samplesPerPixelBox.currentTextChanged.connect(
            self.onSamplesPerPixelChanged
        )

    def onMinErrLineChanged(self):
        try:
            self.model.setMinErrOnCustomImage(self.view.minErrLine.text())
        except Exception as err:
            self.view.minErrLine.setText(str(DEFAULT_MIN_ERR_ON_CUSTOM_IMG))
            self.view.showInfoMessageBox(str(err))

    def onSamplesPerPixelChanged(self, text: str):
        try:
            self.model.setSamplesPerPixelOnCustomImage(text)
            if self.model.samplesPerPixelOnCustomImage > 16:
                msg = "Beware that higher samples per pixel rate during "
                msg += "optimization may cause to unwanted crashes because of "
                msg += "GPU Memory(CUDA) limitations."
                self.view.showInfoMessageBox(msg)
        except Exception as err:
            self.view.samplesPerPixelBox.setCurrentText(
                SUPPORTED_SPP_VALUES[0]
            )
            self.view.showInfoMessageBox(str(err))

    def onCustomRefImgBtnChecked(self):
        if not self.view.customRefImgBtn.isChecked():
            return

        self.hideTableColumn("Minimum Error", True)
        self.hideTableColumn("Optimize", False)
        self.model.setMinErrOnCustomImage(self.view.minErrLine.text())
        self.view.configContainer.show()
        self.loadReferenceImage()

    def hideTableColumn(self, columnLabel: str, hide: bool):
        colIdx = self.getColumnIndex(columnLabel)
        self.view.table.setColumnHidden(colIdx, hide)

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
        colIdx = self.getColumnIndex("Optimize")
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

    def onDefaultRefImgBtnChecked(self):
        if not self.view.defaultRefImgBtn.isChecked():
            return

        self.model.resetReferenceImage()
        self.view.configContainer.hide()
        self.hideTableColumn("Optimize", True)
        self.hideTableColumn("Minimum Error", False)

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


def main():

    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            MY_APP_ID
        )

    LOG_FILE.unlink(missing_ok=True)
    logging.basicConfig(
        filename=LOG_FILE, encoding="utf-8", level=logging.INFO
    )

    app = QApplication(sys.argv)

    # Model
    materialOptimizerModel = MaterialOptimizerModel()

    # View
    materialOptimizerView = MaterialOptimizerView()
    materialOptimizerView.show()

    # Controller
    materialOptimizerController = MaterialOptimizerController(
        model=materialOptimizerModel, view=materialOptimizerView
    )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
