"""
Author: Can Hasbay
"""

from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QApplication, QMessageBox, QTableWidget,
    QTableWidgetItem, QWidget, QVBoxLayout, QProgressBar, QPushButton,
    QHBoxLayout)
from PyQt6.QtGui import QAction
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import sys
import mitsuba as mi
import drjit as dr
import re
import warnings
import logging
from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

mi.set_variant('cuda_ad_rgb')

REFLECTANCE_PATTERN: re.Pattern = re.compile(r'.*\.reflectance\.value')
RADIANCE_PATTERN: re.Pattern = re.compile(r'.*\.radiance\.value')

log_file = Path("python\material-optimizer\material-optimizer.log")
log_file.unlink(missing_ok=True)
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.refImage = None
        self.loadMitsubaScene()

    def setScene(self, scene: mi.Scene):
        self.scene = scene

    def loadMitsubaScene(self, fileName=None):
        if fileName is None:
            self.setScene(mi.load_dict(mi.cornell_box()))
        else:
            self.setScene(mi.load_file(fileName, res=256, integrator='prb'))

        if self.refImage is not None:
            self.refImage = None

        self.params = mi.traverse(self.scene)
        self.referenceReflectanceParams = self.createSubsetSceneParams(
            self.params, REFLECTANCE_PATTERN)
        self.referenceParams = dict(self.referenceReflectanceParams)
        self.optimizationParams = {param: {
            'Learning Rate': 0.3, 'Minimum Error': 0.01} for param in self.referenceParams}

    def getModifiedParams(self):
        result = {}
        for refKey, refValue in self.referenceParams.items():
            if self.params[refKey] != refValue:
                result[refKey] = self.params[refKey]
        return result

    def createSubsetSceneParams(self, params: mi.SceneParameters,
                                pattern: re.Pattern) -> dict:
        return {k: mi.Color3f(v)
                for k, v in params.items() if pattern.search(k)}

    def randomizeSRGBParams(self, params: mi.SceneParameters, ref_params: dict, bound: float):
        if bound < 0.0:
            warnings.warn(
                'Negative bound value! Using 1.0 as the bound value.')
            bound = 1.0

        num_of_channels = 3
        rng = mi.PCG32(size=num_of_channels * len(ref_params))
        samples = rng.next_float64() * bound

        for i, key in enumerate(ref_params):
            if type(params[key]) is not mi.Color3f:
                warn_msg = 'Invalid type:' + str(type(params[key]))
                warnings.warn(warn_msg)
                raise ValueError(
                    'Given ref_params dictionary values must have the mi.Color3f type!')
            params[key] = mi.Color3f(
                samples[i*num_of_channels], samples[i*num_of_channels+1], samples[i*num_of_channels+2])

        params.update()

    def optimize(self, it, reflectanceOpt, opts, paramErrors):
        # check all optimization parameters and if defined threshold is
        # achieved stop optimization for that parameter (i.e. pop optimization param)
        minErrors = {sceneParam: optimizationParam['Minimum Error']
                     for sceneParam, optimizationParam in self.optimizationParams.items()}
        for opt in opts:
            for key in list(opt.keys()):
                if key in opt and paramErrors[key][-1] < minErrors[key]:
                    opt.variables.pop(key)
                    logging.info(f'Key {key} is optimized')

        # stop optimization if all optimization variables are empty
        # (i.e. if all optimization params reached a defined threshold)
        if all(map(lambda opt: not opt.variables, opts)):
            return True

        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(self.scene, self.params, seed=it, spp=4)

        # Evaluate the objective function from the current rendered image
        loss = dr.sum(dr.sqr(image - self.refImage)) / len(image)

        # Backpropagate through the rendering process
        dr.backward(loss)

        for opt in opts:
            # Optimizer: take a gradient descent step
            opt.step()
            for key in opt.keys():
                # Post-process the optimized parameters to ensure legal
                # radiance values
                if REFLECTANCE_PATTERN.search(key):
                    opt[key] = dr.clamp(reflectanceOpt[key], 0.0, 1.0)
            # Update the scene state to the new optimized values
            self.params.update(opt)

        # update errors that are being optimized
        logging.info(f"Iteration {it:02d}")
        for key in self.getModifiedParams():
            err = dr.sum(
                dr.sqr(self.referenceParams[key] - self.params[key]))[0]
            paramErrors[key].append(err)
            logging.info(f"\tkey= {key} error= {paramErrors[key][-1]:6f}")

        return False

    def prepareOptimization(self):
        modifiedParams = self.getModifiedParams()
        reflectanceOpt = mi.ad.Adam(
            lr=0.2, params={k: self.params[k] for k in modifiedParams})
        reflectanceOpt.set_learning_rate(
            {sceneParam: optimizationParam['Learning Rate'] for sceneParam, optimizationParam in self.optimizationParams.items()})
        self.params.update(reflectanceOpt)
        opts = [reflectanceOpt]

        # render initial image
        initImg = mi.render(self.scene, spp=256)

        # set initial parameter errors
        paramErrors = {k: [dr.sum(dr.sqr(self.referenceParams[k] - self.params[k]))[0]]
                       for k in modifiedParams}

        # set optimization parameters
        iteration_count = 200
        return reflectanceOpt, opts, initImg, paramErrors, iteration_count

    def render(self):
        return mi.util.convert_to_bitmap(mi.render(self.scene, spp=512))

    def updateSceneParam(self, key, value):
        if self.params[key] == value:
            return

        self.params[key] = value
        self.params.update()

    def setSceneParamsToModifiedParams(self, modifiedParams):
        for key in modifiedParams:
            self.params[key] = modifiedParams[key]
        self.params.update()

    def setReferenceImage(self):
        # remember modified params
        modifiedParams = self.copyModifiedParams()
        self.setSceneParamsToReferenceParams()
        self.refImage = mi.render(self.scene, spp=512)
        self.setSceneParamsToModifiedParams(modifiedParams)

    def copyModifiedParams(self):
        return {k: mi.Color3f(v) for k, v in self.getModifiedParams().items()}

    def setSceneParamsToReferenceParams(self):
        for key in self.getModifiedParams():
            self.params[key] = self.referenceParams[key]
        self.params.update()

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
        self.setGeometry(300, 300, 550, 450)
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

    def showFileDialog(self):
        homeDir = str(Path.home())
        return QFileDialog.getOpenFileName(
            self, 'Import File', homeDir, 'Xml File (*.xml)')[0]

    def showInfoMessageBox(self, text):
        msgBox = QMessageBox()
        msgBox.setText(text)
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
        msgBox.exec()

    def initCentralWidget(self, sceneParams: dict):
        centralWidget = QWidget(self)
        self.centralLayout = QVBoxLayout(centralWidget)

        self.initTable(sceneParams)
        self.initBottomWidget(centralWidget)

        self.centralLayout.addWidget(self.table)
        self.centralLayout.addWidget(self.bottomWidget)
        self.setCentralWidget(centralWidget)

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
        self.table = QTableWidget(rowsLength, columnsLength)
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setVerticalHeaderLabels(sceneParams.keys())

        for row, param in enumerate(sceneParams):
            for col, label in enumerate(columns):
                if label == 'Value':
                    # if REFLECTANCE_PATTERN.search(param):
                    #     self.setReflectanceToCellWidget(
                    #         sceneParams, row, param, col)
                    # else:
                    if type(sceneParams[param][label]) is mi.Color3f:
                        itemContent = self.Color3fToCellString(
                            sceneParams[param][label])
                        item = QTableWidgetItem(itemContent)
                    else:
                        itemContent = 'Not implemented yet'
                else:
                    item = QTableWidgetItem(str(sceneParams[param][label]))
                self.table.setItem(row, col, item)

    @staticmethod
    def Color3fToCellString(color3f: mi.Color3f):
        return str(color3f).translate(
            str.maketrans({'[': None, ']': None}))

    # def setReflectanceToCellWidget(self, sceneParams, row, param, col):
    #     qColor = QtGui.QColor()
    #     qColor.setRgbF(
    #         sceneParams[param].x[0], sceneParams[param].y[0], sceneParams[param].z[0])
    #     item = ColorButton(color=qColor.name())
    #     self.table.setCellWidget(row, col, item)


# class ColorButton(QtWidgets.QPushButton):
#     '''
#     Taken: https://www.pythonguis.com/widgets/qcolorbutton-a-color-selector-tool-for-pyqt/
#     Additional View: Color Picker
#     Custom Qt Widget to show a chosen color.

#     Left-clicking the button shows the color-chooser, while
#     right-clicking resets the color to None (no-color).
#     '''

#     colorChanged = pyqtSignal(object)

#     def __init__(self, *args, color=None, **kwargs):
#         super(ColorButton, self).__init__(*args, **kwargs)

#         self._color = None
#         self._default = color
#         self.pressed.connect(self.onColorPicker)

#         # Set the initial/default state.
#         self.setColor(self._default)

#     def setColor(self, color):
#         if color != self._color:
#             self._color = color
#             self.colorChanged.emit(color)

#         if self._color:
#             self.setStyleSheet("background-color: %s;" % self._color)
#         else:
#             self.setStyleSheet("")

#     def color(self):
#         return self._color

#     def onColorPicker(self):
#         '''
#         Show color-picker dialog to select color.

#         Qt will use the native dialog by default.

#         '''
#         dlg = QtWidgets.QColorDialog(self)
#         if self._color:
#             dlg.setCurrentColor(QtGui.QColor(self._color))

#         if dlg.exec():
#             self.setColor(dlg.currentColor().name())

#     def mousePressEvent(self, e):
#         if e.button() == Qt.MouseButton.RightButton:
#             self.setColor(self._default)

#         return super(ColorButton, self).mousePressEvent(e)


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

        self.show()


class MaterialOptimizerController:
    """Material Optimizer's controller class."""

    def __init__(self, model: MaterialOptimizerModel,
                 view: MaterialOptimizerView):
        self.model = model
        self.view = view
        tableValues = self.combineTableValues()
        self.view.initCentralWidget(tableValues)
        self.connectSignals()

    def loadMitsubaScene(self):
        fileName = self.view.showFileDialog()
        try:
            self.model.loadMitsubaScene(fileName)
            self.updateCentralWidget()
            self.updateSignals()
        except Exception as err:
            self.model.setScene()
            msg = f"Invalid Mitsuba 3 scene file. Mitsuba Error: {err=}"
            self.view.showInfoMessageBox(msg)

    def updateSignals(self):
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)
        self.view.table.cellChanged.connect(self.onCellChanged)

    def updateCentralWidget(self):
        self.view.setCentralWidget(None)
        tableValues = self.combineTableValues()
        self.view.initCentralWidget(tableValues)

    def combineTableValues(self):
        result = {}
        for key in self.model.referenceParams:
            result[key] = {'Value': self.model.referenceParams[key]}
            result[key].update(self.model.optimizationParams[key])
        return result

    def optimizeMaterials(self):
        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()
            currentSceneParams = self.getCurrentSceneParams()
            self.model.setSceneParamsToModifiedParams(currentSceneParams)

        if len(self.model.getModifiedParams()) <= 0:
            self.view.showInfoMessageBox(
                'Please modify a scene parameter before optimization.')
            return

        self.view.optimizeButton.setDisabled(True)
        if self.model.refImage is None:
            self.view.progressBar.setValue(25)
            self.model.setReferenceImage()
            self.view.progressBar.setValue(self.view.progressBar.maximum())
            self.view.progressBar.reset()

        self.view.progressBar.setValue(50)
        reflectanceOpt, opts, initImg, paramErrors, iteration_count = self.model.prepareOptimization()
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(iteration_count):
            self.view.progressBar.setValue(int(it/iteration_count * 100))
            isOptimized = self.model.optimize(
                it, reflectanceOpt, opts, paramErrors)
            if isOptimized:
                self.view.progressBar.setValue(self.view.progressBar.maximum())
                break

        print('\nOptimization complete.')

        sc = MplCanvas(self.view)
        sc.plotOptimizationResults(self.model.refImage, initImg, self.model.render(),
                                   paramErrors)

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
        if type(self.model.params[param]) is mi.Color3f:
            try:
                newValue = self.model.stringToColor3f(newValue)
            except ValueError as err:
                self.view.table.item(row, col).setText(
                    MaterialOptimizerView.Color3fToCellString(self.model.referenceParams[param]))
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

    # Controller
    materialOptimizerController = MaterialOptimizerController(
        model=materialOptimizerModel, view=materialOptimizerView)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
