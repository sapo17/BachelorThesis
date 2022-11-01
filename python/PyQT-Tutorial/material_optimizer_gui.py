"""
Author: Can Hasbay
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QApplication, QMessageBox, QTableWidget,
    QTableWidgetItem, QWidget, QVBoxLayout, QProgressBar, QPushButton,
    QHBoxLayout)
from PyQt6.QtGui import QAction
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import sys
import mitsuba as mi
import drjit as dr
import re
import warnings
import logging

mi.set_variant('cuda_ad_rgb')

REFLECTANCE_PATTERN: re.Pattern = re.compile(r'.*\.reflectance\.value')
RADIANCE_PATTERN: re.Pattern = re.compile(r'.*\.radiance\.value')


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.setScene()

    def setScene(self, scene: mi.Scene = mi.load_dict(mi.cornell_box())):
        self.scene = scene

    def loadMitsubaScene(self, fileName):
        self.setScene(mi.load_file(fileName, res=256, integrator='prb'))

    def getOptimizableSceneParameters(self) -> dict:
        self.params = mi.traverse(self.scene)
        return self.createSubsetSceneParams(self.params, REFLECTANCE_PATTERN)

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

    def optimize(self, it, refImage, referenceParams, modifiedParams, reflectanceOpt, opts, paramErrors, minErrors):
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
            return True

        # Perform a (noisy) differentiable rendering of the scene
        image = mi.render(self.scene, self.params, seed=it, spp=4)

        # Evaluate the objective function from the current rendered image
        loss = dr.sum(dr.sqr(image - refImage)) / len(image)

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
        for key in modifiedParams.keys():
            err = dr.sum(
                dr.sqr(referenceParams[key] - self.params[key]))[0]
            paramErrors[key].append(err)
            logging.info(f"\tkey= {key} error= {paramErrors[key][-1]:6f}")

        return False

    def prepareOptimization(self):
        refImage = mi.render(self.scene, spp=512)

        # reassign selected scene parameters with random values
        referenceReflectanceParams = self.createSubsetSceneParams(
            self.params, REFLECTANCE_PATTERN)

        # save initial parameter values in a dict (reference values)
        referenceParams = dict(referenceReflectanceParams)

        self.randomizeSRGBParams(self.params, referenceReflectanceParams, 1.0)

        # combine all reference scene parameter dict's
        modifiedParams = dict(referenceReflectanceParams)

        reflectanceOpt = mi.ad.Adam(
            lr=1.0, params={k: self.params[k] for k in referenceReflectanceParams})
        self.params.update(reflectanceOpt)
        opts = [reflectanceOpt]

        # render initial image
        initImg = mi.render(self.scene, spp=256)

        # set initial parameter errors
        paramErrors = {k: [dr.sum(dr.sqr(referenceParams[k] - self.params[k]))[0]]
                       for k in modifiedParams}

        # set optimization parameters
        iteration_count = 200
        minErrors = {k: 0.01 for k in referenceReflectanceParams}
        return refImage, referenceParams, modifiedParams, reflectanceOpt, opts, initImg, paramErrors, iteration_count, minErrors


    def render(self):
        return mi.util.convert_to_bitmap(mi.render(self.scene, spp=512))

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
        self.columns = {'Value': 1.0,
                        'Learning Rate': 1.0, 'Minimum Error': 0.1}
        rowsLength = len(sceneParams)
        columnsLength = len(self.columns)
        self.table = QTableWidget(rowsLength, columnsLength)
        self.table.setHorizontalHeaderLabels(self.columns.keys())
        self.table.setVerticalHeaderLabels(sceneParams.keys())

        for row, param in enumerate(sceneParams):
            for col, label in enumerate(self.columns):
                if label == 'Value':
                    item = QTableWidgetItem(str(sceneParams[param]))
                else:
                    item = QTableWidgetItem(str(self.columns[label]))
                self.table.setItem(row, col, item)


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
        self.view.initCentralWidget(self.model.getOptimizableSceneParameters())

        self.connectSignals()

    def loadMitsubaScene(self):
        fileName = self.view.showFileDialog()

        try:
            self.model.loadMitsubaScene(fileName)
            self.updateCentralWidget()

        except Exception as err:
            self.model.setScene()
            msg = f"Invalid Mitsuba 3 scene file. Mitsuba Error: {err=}"
            self.view.showInfoMessageBox(msg)

    def updateCentralWidget(self):
        self.view.setCentralWidget(None)
        self.view.initCentralWidget(self.model.getOptimizableSceneParameters())
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)

    def optimizeMaterials(self):
        if self.view.progressBar.value() is self.view.progressBar.maximum():
            self.view.progressBar.reset()

        self.view.optimizeButton.setDisabled(True)

        self.view.progressBar.setValue(50)
        refImage, referenceParams, modifiedParams, reflectanceOpt, opts, initImg, paramErrors, iteration_count, minErrors = self.model.prepareOptimization()
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(iteration_count):
            self.view.progressBar.setValue(int(it/iteration_count * 100))
            isOptimized = self.model.optimize(
                it, refImage, referenceParams, modifiedParams, reflectanceOpt, opts, paramErrors, minErrors)
            if isOptimized:
                self.view.progressBar.setValue(self.view.progressBar.maximum())
                break

        print('\nOptimization complete.')

        sc = MplCanvas(self.view)
        sc.plotOptimizationResults(refImage, initImg, self.model.render(),
                                   paramErrors)

        self.view.optimizeButton.setDisabled(False)
        self.view.optimizeButton.setText('Restart Optimization')

    def connectSignals(self):
        self.view.importFile.triggered.connect(self.loadMitsubaScene)
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)


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
