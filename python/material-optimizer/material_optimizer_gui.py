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
import re
import logging
from PyQt6 import QtGui, QtWidgets
import ctypes

# Constants
IMAGES_DIR_PATH = 'images/'
SCENES_DIR_PATH = 'scenes/'
MY_APP_ID = u'sapo.material-optimizer.0.1'  # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(MY_APP_ID)
mi.set_variant('cuda_ad_rgb')
REFLECTANCE_PATTERN: re.Pattern = re.compile(r'.*\.reflectance\.value')
RADIANCE_PATTERN: re.Pattern = re.compile(r'.*\.radiance\.value')
LOG_FILE = Path("material-optimizer.log")
LOG_FILE.unlink(missing_ok=True)
logging.basicConfig(filename=LOG_FILE, encoding='utf-8', level=logging.INFO)


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.refImage = None
        self.sceneRes = (256, 256)
        self.loadMitsubaScene()
        self.setSceneParams(self.scene)
        self.setInitialSceneParams(self.sceneParams)
        self.setDefaultOptimizationParams(self.initialSceneParams)

    def setScene(self, fileName: str, sceneRes: tuple):
        self.scene = mi.load_file(
            fileName, resx=sceneRes[0], resy=sceneRes[1], width=sceneRes[0], height=sceneRes[1], resolution=sceneRes, integrator='prb')

    def loadMitsubaScene(self, fileName=None):
        if fileName is None:
            fileName = SCENES_DIR_PATH + 'cbox.xml'

        self.setScene(fileName, self.sceneRes)
        self.fileName = fileName

    def resetReferenceImage(self):
        if self.refImage is not None:
            self.refImage = None

    def getDefaultOptimizationParams(self, params):
        return {param: {
            'Learning Rate': 0.3, 'Minimum Error': 0.01} for param in params}

    def setDefaultOptimizationParams(self, initialParams):
        self.optimizationParams = self.getDefaultOptimizationParams(
            initialParams)

    def setInitialSceneParams(self, params):
        initialReflectanceParams = self.createSubsetSceneParams(
            params, REFLECTANCE_PATTERN)
        self.initialSceneParams = dict(initialReflectanceParams)

    def setSceneParams(self, scene: mi.Scene):
        self.sceneParams = mi.traverse(scene)

    def getModifiedParams(self):
        result = {}
        for refKey, refValue in self.initialSceneParams.items():
            if self.sceneParams[refKey] != refValue:
                result[refKey] = self.sceneParams[refKey]
        return result

    def createSubsetSceneParams(self, params: mi.SceneParameters,
                                pattern: re.Pattern) -> dict:
        return {k: mi.Color3f(v)
                for k, v in params.items() if pattern.search(k)}

    def updateParamErrors(self, params, initialParams, modifiedParams, paramErrors):
        for key in modifiedParams:
            err = dr.sum(
                dr.sqr(initialParams[key] - params[key]))[0]
            paramErrors[key].append(err)
            logging.info(f"\tkey= {key} error= {paramErrors[key][-1]:6f}")

    def updateAfterStep(self, opts, params):
        for opt in opts:
            # Optimizer: take a gradient descent step
            opt.step()
            for key in opt.keys():
                # Post-process the optimized parameters to ensure legal
                # radiance values
                if REFLECTANCE_PATTERN.search(key):
                    opt[key] = dr.clamp(opt[key], 0.0, 1.0)
            # Update the scene state to the new optimized values
            params.update(opt)

    def mse(self, image, refImage):
        return dr.mean(dr.sqr(refImage - image))

    def getMinParamErrors(self, optimizationParams: dict):
        return {sceneParam: optimizationParam['Minimum Error']
                for sceneParam, optimizationParam in optimizationParams.items()}

    def getInitParamErrors(self, modifiedParams):
        return {k: [dr.sum(dr.sqr(self.initialSceneParams[k] - self.sceneParams[k]))[0]]
                for k in modifiedParams}

    def initOptimizersWithCustomValues(self, customParams):
        opt = mi.ad.Adam(
            lr=0.2, params={k: self.sceneParams[k] for k in customParams})
        opt.set_learning_rate(
            {sceneParam: optimizationParam['Learning Rate'] for sceneParam, optimizationParam in self.optimizationParams.items()})
        opts = [opt]
        return opts

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

    def copyModifiedParams(self):
        return {k: mi.Color3f(v) for k, v in self.getModifiedParams().items()}

    def setSceneParamsToInitialParams(self):
        for key in self.getModifiedParams():
            self.sceneParams[key] = self.initialSceneParams[key]
        self.sceneParams.update()

    def stringToColor3f(self, newValue) -> mi.Color3f:
        result = newValue.split(',')
        errMsg = 'Invalid RGB input. Please use the following format: 1.0, 1.0, 1.0'
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
            mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False)
        resultSize = result.size()
        if resultSize != self.sceneRes:
            result = self.adjustImageSize(result, resultSize)
        result = mi.TensorXf(result)
        result = dr.clamp(result, 0.0, 1.0)
        return result

    def adjustImageSize(self, result, resultSize) -> mi.Bitmap:
        aspectRatio = resultSize[0] / resultSize[1]
        newSize = (int(256 * aspectRatio), 256)
        self.setScene(self.fileName, newSize)
        self.setSceneParams(self.scene)
        result = result.resample(newSize)
        return result

    def updateSceneParamsWithOptimizers(self, opts):
        for opt in opts:
            self.sceneParams.update(opt)


class MaterialOptimizerView(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.layout
        self.show()

    def initUI(self):
        self.initWindowProperties()
        self.statusBar()
        self.initMenu()
        # self.colorPickerWindow = None

    def initWindowProperties(self):
        self.setGeometry(1024, 512, 1024, 512)
        self.setWindowTitle('Material Optimizer')

    def initMenu(self):
        # Menu Bar: Import File
        self.importFile = QAction('Import', self)
        self.importFile.setShortcut('Ctrl+I')
        self.importFile.setStatusTip('Import Mitsuba 3 Scene File')

        # Menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.importFile)

    def showFileDialog(self, filterStr: str):
        homeDir = str(Path.home())
        return QFileDialog.getOpenFileName(
            self, 'Import File', homeDir, filterStr)[0]

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
        self.table = self.initTable(sceneParams)
        self.initBottomWidget(centralWidget)

        self.centralLayout.addWidget(self.topWidget)
        self.centralLayout.addWidget(self.table)
        self.centralLayout.addWidget(self.bottomWidget)
        self.setCentralWidget(centralWidget)

    def initTopWidget(self, centralWidget):
        self.defaultRefImgBtn = QRadioButton(
            'Use original scene for reference image')
        self.defaultRefImgBtn.setChecked(True)
        self.customRefImgBtn = QRadioButton('Use custom reference image')
        self.topWidget = QWidget(centralWidget)
        self.radioBtnLayout = QHBoxLayout(self.topWidget)
        self.radioBtnLayout.addWidget(self.defaultRefImgBtn)
        self.radioBtnLayout.addWidget(self.customRefImgBtn)

    def initBottomWidget(self, centralWidget):
        self.bottomWidget = QWidget(centralWidget)
        self.bottomLayout = QHBoxLayout(self.bottomWidget)
        self.progressBar = QProgressBar(self.bottomWidget)
        self.optimizeButton = QPushButton('Start Optimization', centralWidget)
        self.bottomLayout.addWidget(self.progressBar)
        self.bottomLayout.addWidget(self.optimizeButton)

    def initTable(self, sceneParams: dict):
        firstKey = list(sceneParams)[0]
        columns = sceneParams[firstKey].keys()
        rowsLength = len(sceneParams)
        columnsLength = len(columns)
        result = QTableWidget(rowsLength, columnsLength)
        result.setHorizontalHeaderLabels(columns)
        result.setVerticalHeaderLabels(sceneParams.keys())
        result.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        for row, param in enumerate(sceneParams):
            for col, label in enumerate(columns):
                if label == 'Value':
                    if type(sceneParams[param][label]) is mi.Color3f:
                        itemContent = self.Color3fToCellString(
                            sceneParams[param][label])
                        item = QTableWidgetItem(itemContent)
                    else:
                        itemContent = 'Not implemented yet'
                else:
                    item = QTableWidgetItem(str(sceneParams[param][label]))
                result.setItem(row, col, item)

        return result

    @staticmethod
    def Color3fToCellString(color3f: mi.Color3f):
        return str(color3f).translate(
            str.maketrans({'[': None, ']': None}))

    def replaceTable(self, newTable: QTableWidget):
        self.centralLayout.replaceWidget(self.table, newTable)
        self.table = newTable

# Additional View for Plots


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
        super(MplCanvas, self).__init__(fig)

    def plotOptimizationResults(self, refImage, initImg, finalImg, paramErrors):

        for k, v in paramErrors.items():
            self.axes[0][0].plot(v, label=k)

        self.axes[0][0].set_xlabel('iteration')
        self.axes[0][0].set_ylabel('Loss')
        self.axes[0][0].legend()
        self.axes[0][0].set_title('Parameter error plot')

        self.axes[0][1].imshow(mi.util.convert_to_bitmap(initImg))
        self.axes[0][1].axis('off')
        self.axes[0][1].set_title('Initial Image')

        self.axes[1][0].imshow(mi.util.convert_to_bitmap(finalImg))
        self.axes[1][0].axis('off')
        self.axes[1][0].set_title('Optimized image')

        self.axes[1][1].imshow(mi.util.convert_to_bitmap(refImage))
        self.axes[1][1].axis('off')
        self.axes[1][1].set_title('Reference Image')
        plt.savefig(IMAGES_DIR_PATH + 'material-optimizer-result-figure.png')

        self.show()


class MaterialOptimizerController:
    """Material Optimizer's controller class."""

    def __init__(self, model: MaterialOptimizerModel,
                 view: MaterialOptimizerView):
        self.model = model
        self.view = view
        tableValues = self.combineTableValues(
            self.model.initialSceneParams, self.model.optimizationParams)
        self.view.initCentralWidget(tableValues)
        self.connectSignals()

    def loadMitsubaScene(self):
        try:
            fileName = self.view.showFileDialog('Xml File (*.xml)')
            self.model.loadMitsubaScene(fileName)
            self.model.resetReferenceImage()
            self.view.defaultRefImgBtn.setChecked(True)
            self.model.setSceneParams(self.model.scene)
            self.model.setInitialSceneParams(self.model.sceneParams)
            self.model.setDefaultOptimizationParams(
                self.model.initialSceneParams)
            self.updateTable(self.model.initialSceneParams,
                             self.model.optimizationParams)
        except Exception as err:
            self.model.loadMitsubaScene()
            msg = "Invalid Mitsuba 3 scene file. Setting default scene."
            msg += f" Mitsuba Error: {err=}"
            self.view.showInfoMessageBox(msg)

    def updateTable(self, params, optimizationParams):
        newTable = self.view.initTable(
            self.combineTableValues(params, optimizationParams))
        self.view.replaceTable(newTable)
        self.updateSignals()

    def loadReferenceImage(self):
        try:
            refImgFileName = self.view.showFileDialog('Images (*.png *.jpg)')
            readImg = self.model.readImage(refImgFileName)
            self.model.refImage = readImg
            msg = 'Selected image size is ' + str(readImg.shape)
            msg += 'Resampling the reference image according to the loaded'
            msg += ' scene resolution, if necessary.'
        except Exception as err:
            logging.error(str(err))
            errMsg = 'Cannot load the reference image. Switching back to the'
            errMsg += ' default reference image.'
            self.view.defaultRefImgBtn.setChecked(True)
            self.view.showInfoMessageBox(errMsg)

    def updateSignals(self):
        self.view.table.cellChanged.connect(self.onCellChanged)

    def combineTableValues(self, params, optimizationParams):
        result = {}
        for key in params:
            result[key] = {'Value': params[key]}
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
                'Please modify a scene parameter before optimization.')
            return

        self.view.optimizeButton.setDisabled(True)
        if self.model.refImage is None:
            self.view.progressBar.setValue(25)
            self.model.setReferenceImageToInitiallyLoadedScene()
            self.view.progressBar.setValue(self.view.progressBar.maximum())
            self.view.progressBar.reset()

        self.view.progressBar.setValue(50)
        opts = self.model.initOptimizersWithCustomValues(modifiedParams)
        self.model.updateSceneParamsWithOptimizers(opts)
        initImg = self.model.render(self.model.scene, spp=256)
        paramErrors = self.model.getInitParamErrors(modifiedParams)
        minErrors = self.model.getMinParamErrors(self.model.optimizationParams)
        iterationCount = 200
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(iterationCount):
            self.view.progressBar.setValue(int(it/iterationCount * 100))

            # check all optimization parameters and if defined threshold is
            # achieved stop optimization for that parameter (i.e. pop optimization param)
            for opt in opts:
                for key in list(opt.keys()):
                    if key in opt and paramErrors[key][-1] < minErrors[key]:
                        opt.variables.pop(key)
                        logging.info(f'Key {key} is optimized')

            # stop optimization if all optimization variables are empty
            # (i.e. if all optimization params reached a defined threshold)
            if all(map(lambda opt: not opt.variables, opts)):
                break

            # Perform a (noisy) differentiable rendering of the scene
            image = self.model.render(
                self.model.scene, spp=4, params=self.model.sceneParams, seed=it)

            # Evaluate the objective function from the current rendered image
            loss = self.model.mse(image, self.model.refImage)
            # loss = dr.sum(dr.sqr(image - self.refImage)) / len(image)

            # Backpropagate through the rendering process
            dr.backward(loss)

            self.model.updateAfterStep(opts, self.model.sceneParams)

            logging.info(f"Iteration {it:02d}")
            self.model.updateParamErrors(
                self.model.sceneParams, self.model.initialSceneParams, modifiedParams, paramErrors)

        self.view.progressBar.setValue(self.view.progressBar.maximum())
        sc = MplCanvas(self.view)
        sc.plotOptimizationResults(self.model.refImage, initImg, mi.util.convert_to_bitmap(self.model.render(self.model.scene, 512)),
                                   paramErrors)
        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText('Restart Optimization')

    def optimizeMaterialsWithCustomReferenceImage(self):
        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()
            currentSceneParams = self.getCurrentSceneParams()
            self.model.updateSceneParameters(currentSceneParams)

        self.view.optimizeButton.setDisabled(True)
        self.view.progressBar.setValue(50)
        opts = self.model.initOptimizersWithCustomValues(
            self.model.createSubsetSceneParams(self.model.sceneParams, REFLECTANCE_PATTERN))
        self.model.updateSceneParamsWithOptimizers(opts)
        initImg = self.model.render(self.model.scene, spp=256)
        iterationCount = 100
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        lossHistKey = 'MSE(Current-Image, Reference-Image)'
        lossHist = {lossHistKey: []}
        for it in range(iterationCount):
            self.view.progressBar.setValue(int(it/iterationCount * 100))

            # Perform a (noisy) differentiable rendering of the scene
            image = self.model.render(
                self.model.scene, spp=4, params=self.model.sceneParams, seed=it)
            image = dr.clamp(image, 0.0, 1.0)

            # Evaluate the objective function from the current rendered image
            loss = self.model.mse(image, self.model.refImage)
            lossHist[lossHistKey].append(loss)
            logging.info(f"Iteration {it:02d}")
            logging.info(f"\tcurrent loss= {loss[0]:6f}")

            minLossErr = 0.00095
            if loss[0] < minLossErr:
                break

            # Backpropagate through the rendering process
            dr.backward(loss)

            self.model.updateAfterStep(opts, self.model.sceneParams)

        self.view.progressBar.setValue(self.view.progressBar.maximum())
        sc = MplCanvas(self.view)
        sc.plotOptimizationResults(self.model.refImage, initImg, mi.util.convert_to_bitmap(
            self.model.render(self.model.scene, 512)), lossHist)
        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText('Restart Optimization')

    def getCurrentSceneParams(self):
        currentSceneParams = {}
        for row in range(self.view.table.rowCount()):
            key = self.view.table.verticalHeaderItem(row).text()
            # since cell values for scene param cannot be other than
            # mi.Color3f, no check for type casting is necessary
            # see also self.onSceneParamChanged()
            currentSceneParams[key] = self.model.stringToColor3f(
                self.view.table.item(row, 0).text())
        return currentSceneParams

    def connectSignals(self):
        self.view.importFile.triggered.connect(self.loadMitsubaScene)
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)
        self.view.table.cellChanged.connect(self.onCellChanged)
        self.view.customRefImgBtn.toggled.connect(
            self.onCustomRefImgBtnChecked)
        self.view.defaultRefImgBtn.toggled.connect(
            self.onDefaultRefImgBtnChecked)

    def onCustomRefImgBtnChecked(self):
        if not self.view.customRefImgBtn.isChecked():
            return

        self.loadReferenceImage()
        self.hideTableColumn('Minimum Error', True)

    def hideTableColumn(self, columnLabel: str, hide: bool):
        for colIdx in range(self.view.table.columnCount()):
            if self.view.table.horizontalHeaderItem(colIdx).text() == columnLabel:
                break
        self.view.table.setColumnHidden(colIdx, hide)

    def onDefaultRefImgBtnChecked(self):
        if not self.view.defaultRefImgBtn.isChecked():
            return

        self.model.resetReferenceImage()
        self.hideTableColumn('Minimum Error', False)

    def onCellChanged(self, row, col):
        if col == 0:
            isSuccess, param, newValue = self.onSceneParamChanged(row, col)
            if isSuccess:
                self.model.updateSceneParam(param, newValue)
        else:
            isSuccess, paramRow, paramCol, newValue = self.onOptimizationParamChanged(
                row, col)
            if isSuccess:
                self.model.updateOptimizationParam(
                    paramCol, paramRow, newValue)

    def onOptimizationParamChanged(self, row, col):
        paramRow = self.view.table.verticalHeaderItem(
            self.view.table.currentRow()).text()
        paramCol = self.view.table.horizontalHeaderItem(
            self.view.table.currentColumn()).text()
        try:
            newValue = float(self.view.table.item(row, col).text())
        except:
            self.view.table.item(row, col).setText(
                str(self.model.optimizationParams[paramRow][paramCol]))
            errMsg = 'Optimization parameter is not changed.'
            self.view.showInfoMessageBox(errMsg)
            return False, None, None, None
        return True, paramRow, paramCol, newValue

    def onSceneParamChanged(self, row, col):
        param = self.view.table.verticalHeaderItem(
            self.view.table.currentRow()).text()
        newValue = self.view.table.item(row, col).text()
        if type(self.model.sceneParams[param]) is mi.Color3f:
            try:
                newValue = self.model.stringToColor3f(newValue)
            except ValueError as err:
                self.view.table.item(row, col).setText(
                    MaterialOptimizerView.Color3fToCellString(self.model.initialSceneParams[param]))
                errMsg = 'Scene parameter is not changed. ' + str(err)
                self.view.showInfoMessageBox(errMsg)
                return False, None, None
        return True, param, newValue


def main():

    app = QApplication(sys.argv)

    # Model
    materialOptimizerModel = MaterialOptimizerModel()

    # View
    materialOptimizerView = MaterialOptimizerView()
    materialOptimizerView.show()
    materialOptimizerView.setWindowIcon(
        QtGui.QIcon(IMAGES_DIR_PATH + 'sloth.png'))

    # Controller
    materialOptimizerController = MaterialOptimizerController(
        model=materialOptimizerModel, view=materialOptimizerView)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
