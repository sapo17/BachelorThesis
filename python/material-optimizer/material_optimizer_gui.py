"""
Author: Can Hasbay
"""

import os
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
import json
import datetime
import numpy as np

mi.set_variant(CUDA_AD_RGB)


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.refImage = None
        self.sceneRes = (256, 256)
        self.loadMitsubaScene()
        self.setSceneParams(self.scene)
        self.setInitialSceneParams(self.sceneParams)
        self.setDefaultOptimizationParams(self.initialSceneParams)
        self.setSamplesPerPixelOnCustomImage(SUPPORTED_SPP_VALUES[0])
        self.setLossFunctionOnCustomImage(LOSS_FUNCTION_STRINGS[0])

    def setScene(self, fileName: str, sceneRes: tuple, integratorType: str):
        self.scene = mi.load_file(
            fileName,
            resx=sceneRes[0],
            resy=sceneRes[1],
            integrator=integratorType,
        )

    def loadMitsubaScene(self, fileName=None):
        if fileName is None:
            fileName = SCENES_DIR_PATH + DEFAULT_MITSUBA_SCENE
            self.handleMissingCboxFile(fileName)

        self.integratorType = self.findPossibleIntegratorType(fileName)
        self.setScene(fileName, self.sceneRes, self.integratorType)
        self.fileName = fileName

    def handleMissingCboxFile(self, fileName):
        if not os.path.isfile(fileName):
            self.createDirectoryIfNotExists(SCENES_DIR_PATH)
            self.createDirectoryIfNotExists(SCENES_MESHES_DIR_PATH)
            self.createObjFileIfNotExists(
                CBOX_LUMINAIRE_OBJ_PATH, CBOX_LUMINAIRE_OBJ_STRING
            )
            self.createObjFileIfNotExists(
                CBOX_FLOOR_OBJ_PATH, CBOX_FLOOR_OBJ_STRING
            )
            self.createObjFileIfNotExists(
                CBOX_CEILING_OBJ_PATH, CBOX_CEILING_OBJ_STRING
            )
            self.createObjFileIfNotExists(
                CBOX_BACK_OBJ_PATH, CBOX_BACK_OBJ_STRING
            )
            self.createObjFileIfNotExists(
                CBOX_GREENWALL_OBJ_PATH, CBOX_GREENWALL_OBJ_STRING
            )
            self.createObjFileIfNotExists(
                CBOX_REDWALL_OBJ_PATH, CBOX_REDWALL_OBJ_STRING
            )
            self.createObjFileIfNotExists(CBOX_SCENE_PATH, CBOX_XML_STRING)

    def createDirectoryIfNotExists(self, dirPath: str):
        if not os.path.exists(dirPath):
            msg = f"Missing directory on {dirPath}!"
            logging.error(msg)
            os.makedirs(dirPath)

    def createObjFileIfNotExists(self, path: str, content: str):
        if not os.path.isfile(path):
            msg = f"Missing file on {path}!"
            logging.error(msg)
            with open(path, "w") as f:
                msg = f"Created: {path}"
                logging.info(msg)
                f.write(content)

    def findPossibleIntegratorType(self, fileName) -> str:
        tmpScene = mi.load_file(fileName)
        tmpParams = mi.traverse(tmpScene)
        integratorType = MITSUBA_PRB_INTEGRATOR
        if self.anyInPatterns(
            tmpParams, PATTERNS_REQUIRE_VOLUMETRIC_INTEGRATOR
        ):
            if self.anyInPatterns(
                tmpParams, PATTERNS_INTRODUCE_DISCONTINUITIES
            ):
                msg = f"Beware that {MITSUBA_PRBVOLPATH_INTEGRATOR} is in use,"
                msg += " however the scene contains some parameters that may "
                msg += (
                    "introduce discontinuities! Information from the mitsuba"
                )
                msg += (
                    " documentation: No reparameterization. This means that "
                )
                msg += " the integrator cannot be used for shape optimization "
                msg += "(it will return incorrect/biased gradients for "
                msg += "geometric parameters like vertex positions.)"
                logging.warning(msg)
            integratorType = MITSUBA_PRBVOLPATH_INTEGRATOR
        elif self.anyInPatterns(tmpParams, PATTERNS_INTRODUCE_DISCONTINUITIES):
            integratorType = MITSUBA_PRB_REPARAM_INTEGRATOR

        logging.info(f"Integrator type: {integratorType}")
        return integratorType

    def anyInPatterns(self, tmpParams, patterns: list):
        return any(
            pattern.search(k)
            for k in tmpParams.properties
            for pattern in patterns
        )

    def resetReferenceImage(self):
        if self.refImage is not None:
            self.refImage = None

    def getDefaultOptimizationParams(self, params):
        return {
            param: {
                COLUMN_LABEL_LEARNING_RATE: 0.3,
                COLUMN_LABEL_MINIMUM_ERROR: 0.01,
                COLUMN_LABEL_OPTIMIZE: False,
            }
            for param in params
        }

    def setDefaultOptimizationParams(self, initialParams):
        self.optimizationParams = self.getDefaultOptimizationParams(
            initialParams
        )

    def setInitialSceneParams(self, params):
        self.initialSceneParams = dict(
            self.createSubsetSceneParams(
                params, SUPPORTED_MITSUBA_PARAMETER_PATTERNS
            )
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
            else:
                result[k] = mi.Float(v)
        elif vType is mi.TensorXf:
            result[k] = mi.TensorXf(v)

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
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_ETA_VALUE
            )
        if K_PATTERN.search(key):
            opt[key] = dr.clamp(opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_K_VALUE)
        if ALPHA_PATTERN.search(key):
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_ALPHA_VALUE
            )
        elif DIFF_TRANS_PATTERN.search(key):
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_DIFF_TRANS_VALUE
            )
        elif DELTA_PATTERN.search(key):
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_DELTA_VALUE
            )
        elif PHASE_G_PATTERN.search(key):
            opt[key] = dr.clamp(opt[key], MIN_PHASE_G_VALUE, MAX_PHASE_G_VALUE)
        elif SCALE_PATTERN.search(key):
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_SCALE_VALUE
            )
        else:
            opt[key] = dr.clamp(
                opt[key], DEFAULT_MIN_CLAMP_VALUE, DEFAULT_MAX_CLAMP_VALUE
            )

    def getMinParamErrors(self, optimizationParams: dict):
        return {
            sceneParam: optimizationParam[COLUMN_LABEL_MINIMUM_ERROR]
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
                k: self.optimizationParams[k][COLUMN_LABEL_LEARNING_RATE]
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

    def setIterationCountOnCustomImage(self, value: str):
        if not MaterialOptimizerModel.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 50)")
        self.iterationCountOnCustomImage = int(value)

    def setSamplesPerPixelOnCustomImage(self, value: str):
        if not MaterialOptimizerModel.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 4)")
        self.samplesPerPixelOnCustomImage = int(value)

    def setLossFunctionOnCustomImage(self, value: str):
        self.lossFunctionOnCustomImage = value

    @staticmethod
    def minIdxInDrList(lis: list):
        minValue = dr.min(lis)
        return lis.index(minValue)

    def computeLoss(self, seed: int = 0):
        # Perform a (noisy) differentiable rendering of the scene
        image = self.render(
            self.scene,
            spp=self.samplesPerPixelOnCustomImage,
            params=self.sceneParams,
            seed=seed,
        )
        image = dr.clamp(image, 0.0, 1.0)

        # Evaluate the objective function from the current rendered image
        if self.lossFunctionOnCustomImage == MSE_STRING:
            result = self.mse(image, self.refImage)
        elif self.lossFunctionOnCustomImage == BRIGHTNESS_IDP_MSE_STRING:
            result = self.brightnessIndependentMSE(image, self.refImage)
        elif self.lossFunctionOnCustomImage == DUAL_BUFFER_STRING:
            image2 = self.render(
                self.scene,
                spp=self.samplesPerPixelOnCustomImage,
                params=self.sceneParams,
                seed=seed + 1,
            )
            image2 = dr.clamp(image2, 0.0, 1.0)
            result = self.dualBufferError(image, image2)
        elif self.lossFunctionOnCustomImage == MAE_STRING:
            result = self.mae(image, self.refImage)
        elif self.lossFunctionOnCustomImage == MBE_STRING:
            result = self.mbe(image, self.refImage)
        else:
            result = self.mse(image, self.refImage)

        return result

    def mse(self, image, refImage):
        """L2 Loss: Mean Squared Error"""
        return dr.mean(dr.sqr(refImage - image))

    def brightnessIndependentMSE(self, image, ref):
        """
        Brightness-independent L2 loss function.
        Taken from: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html#6.-Running-the-optimization
        """
        scaled_image = image / dr.mean(dr.detach(image))
        scaled_ref = ref / dr.mean(ref)
        return dr.mean(dr.sqr(scaled_image - scaled_ref))

    def dualBufferError(self, image, image2):
        """
        Loss Function mentioned in: Reconstructing Translucent Objects Using Differentiable Rendering, Deng et al.
        @inproceedings{10.1145/3528233.3530714,
        author = {Deng, Xi and Luan, Fujun and Walter, Bruce and Bala, Kavita and Marschner, Steve},
        title = {Reconstructing Translucent Objects Using Differentiable Rendering},
        year = {2022},
        isbn = {9781450393379},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3528233.3530714},
        doi = {10.1145/3528233.3530714},
        abstract = {Inverse rendering is a powerful approach to modeling objects from photographs, and we extend previous techniques to handle translucent materials that exhibit subsurface scattering. Representing translucency using a heterogeneous bidirectional scattering-surface reflectance distribution function (BSSRDF), we extend the framework of path-space differentiable rendering to accommodate both surface and subsurface reflection. This introduces new types of paths requiring new methods for sampling moving discontinuities in material space that arise from visibility and moving geometry. We use this differentiable rendering method in an end-to-end approach that jointly recovers heterogeneous translucent materials (represented by a BSSRDF) and detailed geometry of an object (represented by a mesh) from a sparse set of measured 2D images in a coarse-to-fine framework incorporating Laplacian preconditioning for the geometry. To efficiently optimize our models in the presence of the Monte Carlo noise introduced by the BSSRDF integral, we introduce a dual-buffer method for evaluating the L2 image loss. This efficiently avoids potential bias in gradient estimation due to the correlation of estimates for image pixels and their derivatives and enables correct convergence of the optimizer even when using low sample counts in the renderer. We validate our derivatives by comparing against finite differences and demonstrate the effectiveness of our technique by comparing inverse-rendering performance with previous methods. We show superior reconstruction quality on a set of synthetic and real-world translucent objects as compared to previous methods that model only surface reflection.},
        booktitle = {ACM SIGGRAPH 2022 Conference Proceedings},
        articleno = {38},
        numpages = {10},
        keywords = {differentiable rendering, ray tracing, appearance acquisition, subsurface scattering},
        location = {Vancouver, BC, Canada},
        series = {SIGGRAPH '22}
        }
        """
        return dr.mean((image - self.refImage) * (image2 - self.refImage))

    def mae(self, image, refImage):
        """L1 Loss: Mean Absolute Error"""
        return dr.mean(dr.abs(refImage - image))

    def mbe(self, image, refImage):
        """Mean Bias Error"""
        return dr.mean(refImage - image)


class MaterialOptimizerView(QMainWindow):
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
        # self.colorPickerWindow = None

    def initWindowProperties(self):
        self.setGeometry(1024, 512, 1024, 512)
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

    def showFileDialog(self, filterStr: str):
        homeDir = str(Path.home())
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
        minErrLabel = QLabel(text=COLUMN_LABEL_MINIMUM_ERROR)
        self.minErrLine = QLineEdit()
        self.minErrLine.setText(str(DEFAULT_MIN_ERR_ON_CUSTOM_IMG))
        self.minErrContainerLayout.addWidget(minErrLabel)
        self.minErrContainerLayout.addWidget(self.minErrLine)

        # samples per pixel dropdown
        self.sppContainer = QWidget(self.configContainer)
        self.sppContainerLayout = QHBoxLayout(self.sppContainer)
        samplesPerPixelLabel = QLabel(text=SPP_DURING_OPT_STRING)
        self.samplesPerPixelBox = QComboBox()
        self.samplesPerPixelBox.addItems(SUPPORTED_SPP_VALUES)
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
        self.iterationCountLine.setText(
            str(DEFAULT_ITERATION_COUNT_ON_CUSTOM_IMG)
        )
        self.iterationContainerLayout.addWidget(iterationCountLabel)
        self.iterationContainerLayout.addWidget(self.iterationCountLine)

        self.configContainerLayout.addWidget(self.minErrContainer)
        self.configContainerLayout.addWidget(self.sppContainer)
        self.configContainerLayout.addWidget(self.lossFunctionContainer)
        self.configContainerLayout.addWidget(self.iterationContainer)
        self.configContainer.hide()

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

    def initOptimizedSceneSelector(
        self,
        model: MaterialOptimizerModel,
        initImg,
        lossHist: list,
        sceneParamsHist: list,
    ):
        self.model = model
        self.initImg = initImg
        self.lossHist = lossHist
        self.sceneParamsHist = sceneParamsHist

        # central widget
        centralWidgetContainer = QWidget()
        centralWidgetContainerLayout = QVBoxLayout(centralWidgetContainer)

        # dropdown menu
        self.comboBox = QComboBox()
        self.comboBox.addItems(
            [LAST_ITERATION_STRING, COLUMN_LABEL_MINIMUM_ERROR]
        )
        self.comboBox.currentTextChanged.connect(
            self.onOptimizedSceneSelectorTextChanged
        )

        # output button
        outputBtn = QPushButton(text=OUTPUT_TO_JSON_STRING)
        outputBtn.clicked.connect(self.onOutputBtnPressed)

        centralWidgetContainerLayout.addWidget(self.comboBox)
        centralWidgetContainerLayout.addWidget(outputBtn)
        self.setCentralWidget(centralWidgetContainer)

        self.show()
        lastIteration = len(self.sceneParamsHist) - 1
        self.showOptimizedPlot(lastIteration)

    def onOutputBtnPressed(self):
        selectedIteration = len(self.sceneParamsHist) - 1
        if self.comboBox.currentText() == COLUMN_LABEL_MINIMUM_ERROR:
            selectedIteration = MaterialOptimizerModel.minIdxInDrList(
                self.lossHist
            )

        outputFileName = f"{OUTPUT_DIR_PATH}scene_paramaters_iteration_{selectedIteration}_{datetime.datetime.now().isoformat('_', 'seconds')}.json"
        outputFileName = outputFileName.replace(":", "_")
        with open(outputFileName, "w") as outfile:
            outputDict = {}
            for k, v in self.sceneParamsHist[selectedIteration].items():
                if type(v) is mi.TensorXf:
                    if ALBEDO_DATA_PATTERN.search(k):
                        outputVolumeFileName = f"{OUTPUT_DIR_PATH}volume_{k}_iteration_{selectedIteration}_{datetime.datetime.now().isoformat('_', 'seconds')}.vol"
                        outputVolumeFileName = outputVolumeFileName.replace(
                            ":", "_"
                        )
                        mi.VolumeGrid(v).write(outputVolumeFileName)
                    else:
                        outputTextureFileName = f"{OUTPUT_DIR_PATH}texture_{k}_iteration_{selectedIteration}_{datetime.datetime.now().isoformat('_', 'seconds')}.png"
                        outputTextureFileName = outputTextureFileName.replace(
                            ":", "_"
                        )
                        mi.util.write_bitmap(outputTextureFileName, v)
                elif type(v) is mi.Float:
                    floatArray = [f for f in v]
                    if VERTEX_COLOR_PATTERN.search(k):
                        outputVertexColorFileName = f"{OUTPUT_DIR_PATH}vertex_color_numpy_array_{k}_iteration_{selectedIteration}_{datetime.datetime.now().isoformat('_', 'seconds')}.npy"
                        outputVertexColorFileName = outputVertexColorFileName.replace(
                            ":", "_"
                        )
                        np.save(outputVertexColorFileName, np.array(floatArray))
                    else:
                        outputDict[k] = floatArray
                else:
                    outputDict[k] = str(v)
            json.dump(outputDict, outfile, indent=4)

    def onOptimizedSceneSelectorTextChanged(self, text: str):
        if text == COLUMN_LABEL_MINIMUM_ERROR:
            self.showOptimizedPlot(
                MaterialOptimizerModel.minIdxInDrList(self.lossHist)
            )
        else:
            self.showOptimizedPlot(len(self.sceneParamsHist) - 1)

    def showOptimizedPlot(self, iteration: int):
        logging.info(
            f"Scene parameters in {iteration}:\n {self.sceneParamsHist[iteration]}"
        )
        self.model.sceneParams.update(values=self.sceneParamsHist[iteration])
        sc = MplCanvas()
        lossHistKey = self.model.lossFunctionOnCustomImage
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

        self.axes[0][0].set_xlabel(ITERATION_STRING)
        self.axes[0][0].set_ylabel(LOSS_STRING)
        self.axes[0][0].legend()
        self.axes[0][0].set_title(PARAMETER_ERROR_PLOT_STRING)

        self.axes[0][1].imshow(mi.util.convert_to_bitmap(initImg))
        self.axes[0][1].axis(OFF_STRING)
        self.axes[0][1].set_title(INITIAL_IMAGE_STRING)

        self.axes[1][0].imshow(mi.util.convert_to_bitmap(finalImg))
        self.axes[1][0].axis(OFF_STRING)
        self.axes[1][0].set_title(
            f"Optimized image: Iteration #{iterationNumber}, Loss: {loss:6f}"
        )

        self.axes[1][1].imshow(mi.util.convert_to_bitmap(refImage))
        self.axes[1][1].axis(OFF_STRING)
        self.axes[1][1].set_title(REFERENCE_IMAGE_STRING)
        plt.savefig(IMAGES_DIR_PATH + FIGURE_FILE_NAME)

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
        self.hideTableColumn(COLUMN_LABEL_OPTIMIZE, True)
        self.connectSignals()

    def loadMitsubaScene(self):
        try:
            fileName = self.view.showFileDialog(XML_FILE_FILTER_STRING)
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
        self.hideTableColumn(COLUMN_LABEL_OPTIMIZE, True)
        self.updateSignals()

    def loadReferenceImage(self):
        try:
            refImgFileName = self.view.showFileDialog(
                IMAGES_FILE_FILTER_STRING
            )
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
            result[key] = {COLUMN_LABEL_VALUE: params[key]}
            result[key].update(optimizationParams[key])
        return result

    def optimizeMaterials(self):
        try:
            if self.view.defaultRefImgBtn.isChecked():
                self.optimizeMaterialsWithDefaultReferenceImage()
            else:
                self.optimizeMaterialsWithCustomReferenceImage()
        except Exception as err:
            msg = f"Exiting program. Runtime error during optimiztion: {err}"
            logging.error(msg)
            self.view.showInfoMessageBox(msg)
            sys.exit()

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
                    self.model.sceneParams,
                    SUPPORTED_MITSUBA_PARAMETER_PATTERNS,
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
        self.view.optimizeButton.setText(RESTART_OPTIMIZATION_STRING)

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
        sceneParamsHist = []
        lossHist = []
        self.view.progressBar.setValue(100)

        self.view.progressBar.reset()
        for it in range(self.model.iterationCountOnCustomImage):
            self.view.progressBar.setValue(
                int(it / self.model.iterationCountOnCustomImage * 100)
            )

            loss = self.model.computeLoss(it)
            lossHist.append(loss)
            sceneParamsHist.append(
                self.model.createSubsetSceneParams(
                    self.model.sceneParams,
                    SUPPORTED_MITSUBA_PARAMETER_PATTERNS,
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
        logging.info(f"Initial scene parameters:\n {sceneParamsHist[0]}")

        if it <= 0:
            self.view.showInfoMessageBox("No optimization was necessary")
        else:
            popUp = PopUpWindow(self.view)
            popUp.initOptimizedSceneSelector(
                self.model, initImg, lossHist, sceneParamsHist
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
        self.view.optimizeButton.clicked.connect(self.optimizeMaterials)
        self.view.table.cellChanged.connect(self.onCellChanged)
        self.view.customRefImgBtn.toggled.connect(
            self.onCustomRefImgBtnChecked
        )
        self.view.defaultRefImgBtn.toggled.connect(
            self.onDefaultRefImgBtnChecked
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
            self.model.setMinErrOnCustomImage(self.view.minErrLine.text())
        except Exception as err:
            self.view.minErrLine.setText(str(DEFAULT_MIN_ERR_ON_CUSTOM_IMG))
            self.view.showInfoMessageBox(str(err))

    def onIterationCountLineChanged(self):
        try:
            self.model.setIterationCountOnCustomImage(
                self.view.iterationCountLine.text()
            )
        except Exception as err:
            self.view.iterationCountLine.setText(
                str(DEFAULT_ITERATION_COUNT_ON_CUSTOM_IMG)
            )
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

    def onLossFunctionChanged(self, text: str):
        self.model.setLossFunctionOnCustomImage(text)

    def onCustomRefImgBtnChecked(self):
        if not self.view.customRefImgBtn.isChecked():
            return

        self.hideTableColumn(COLUMN_LABEL_MINIMUM_ERROR, True)
        self.hideTableColumn(COLUMN_LABEL_OPTIMIZE, False)
        self.model.setMinErrOnCustomImage(self.view.minErrLine.text())
        self.model.setIterationCountOnCustomImage(
            self.view.iterationCountLine.text()
        )
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

    def onDefaultRefImgBtnChecked(self):
        if not self.view.defaultRefImgBtn.isChecked():
            return

        self.model.resetReferenceImage()
        self.view.configContainer.hide()
        self.hideTableColumn(COLUMN_LABEL_OPTIMIZE, True)
        self.hideTableColumn(COLUMN_LABEL_MINIMUM_ERROR, False)

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
