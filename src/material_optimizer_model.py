"""
Author: Can Hasbay
"""

from __future__ import annotations
import os
import mitsuba as mi
import drjit as dr
import logging
from src.constants import *
import time
from abc import ABC, abstractmethod


class MaterialOptimizerModel:
    """This class contains the business logic of material-optimizer."""

    def __init__(self) -> None:
        self.sensorToReferenceImageDict = None
        self.sceneRes = (256, 256)
        self.setOptimizerStrategy(DefaultOptimizerStrategy(self))
        self.loadMitsubaScene()
        self.setSceneParams(self.scene)
        self.setInitialSceneParams(self.sceneParams)
        self.setDefaultOptimizationParams(self.initialSceneParams)
        self.setSamplesPerPixel(SUPPORTED_SPP_VALUES[2])
        self.setLossFunction(LOSS_FUNCTION_STRINGS[0])
        self.setMarginPercentage(INF_STR)
        self.setMarginPenalty(NONE_STR)

    @property
    def getOptimizerStrategy(self) -> OptimizerStrategy:
        return self.optimizerStrategy

    def setOptimizerStrategy(
        self, optimizerStrategy: OptimizerStrategy
    ) -> None:
        self.optimizerStrategy = optimizerStrategy

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
        self.setOptimizerStrategy(self.findPossibleOptimizerStrategy(fileName))
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

    def findPossibleOptimizerStrategy(self, fileName) -> OptimizerStrategy:
        tmpScene: mi.Scene = mi.load_file(fileName)
        tmpParams = mi.traverse(tmpScene)

        result = DefaultOptimizerStrategy(self)
        if self.anyInPatterns(tmpParams, OPTIMIZER_PATTERNS):
            result = CustomOptimizerStrategy(self)
            logging.info(f"Optimizer type: custom")

        return result

    def anyInPatterns(self, tmpParams, patterns: list):
        return any(
            pattern.search(k)
            for k in tmpParams.properties
            for pattern in patterns
        )

    def resetSensorToReferenceImageDict(self):
        if self.sensorToReferenceImageDict is not None:
            self.sensorToReferenceImageDict = None

    def getDefaultOptimizationParams(self, params) -> dict:
        return {
            param: {
                COLUMN_LABEL_LEARNING_RATE: DEFAULT_LEARNING_RATE,
                COLUMN_LABEL_MIN_CLAMP_LABEL: DEFAULT_MIN_CLAMP_VALUE,
                COLUMN_LABEL_MAX_CLAMP_LABEL: DEFAULT_MAX_CLAMP_VALUE,
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

    def updateAfterStep(self, opts, params, it=0):
        for opt in opts:
            # Optimizer: take a gradient descent step
            opt.step()
            for key in opt.keys():
                self.ensureLegalParamValues(opt, key, it)
            # Update the scene state to the new optimized values
            params.update(opt)

    def ensureLegalParamValues(self, opt, key, it):
        # Post-process the optimized parameters to ensure legal values
        if VERTEX_POSITIONS_PATTERN.search(
            key
        ) or VERTEX_NORMALS_PATTERN.search(key):
            self.ensureLegalVertexPositions(opt, key)
        elif GRID_VOLUME_TO_OPTIMIZER_PATTERN.search(key):
            if it > 32:
                opt[key] = dr.select(
                    opt[key] <= dr.min(opt[key])[0] + 0.3 * np.std(opt[key]),
                    0.0,
                    1.0,
                )
            else:
                opt[key] = dr.clamp(
                    opt[key],
                    0.0,
                    self.optimizationParams[key][COLUMN_LABEL_MAX_CLAMP_LABEL],
                )
        else:
            opt[key] = dr.clamp(
                opt[key],
                self.optimizationParams[key][COLUMN_LABEL_MIN_CLAMP_LABEL],
                self.optimizationParams[key][COLUMN_LABEL_MAX_CLAMP_LABEL],
            )

    def ensureLegalVertexPositions(self, opt, key):
        point = dr.unravel(mi.Point3f, opt[key])
        minVal = self.optimizationParams[key][COLUMN_LABEL_MIN_CLAMP_LABEL]
        maxVal = self.optimizationParams[key][COLUMN_LABEL_MAX_CLAMP_LABEL]
        clampedPoint = dr.clamp(point, minVal, maxVal)
        opt[key] = dr.ravel(clampedPoint)

    def getClosestPattern(self, key: str) -> re.Pattern:
        for pattern in SUPPORTED_MITSUBA_PARAMETER_PATTERNS:
            if pattern.search(key):
                return pattern
        return EMPTY_PATTERN

    def initOptimizers(self, params: list) -> list:
        opt = mi.ad.Adam(
            lr=0.2,
            params={k: self.sceneParams[k] for k in params},
            mask_updates=True,
        )
        opt.set_learning_rate(
            {
                k: self.optimizationParams[k][COLUMN_LABEL_LEARNING_RATE]
                for k in params
            }
        )
        return [opt]

    @staticmethod
    def render(
        scene: mi.Scene, sensor: mi.Sensor, spp: int, params=None, seed=0
    ):
        return mi.render(scene, params, sensor=sensor, seed=seed, spp=spp)

    def updateSceneParam(self, key, value):
        if self.sceneParams[key] == value:
            return

        self.sceneParams[key] = value
        self.sceneParams.update()

    def updateSceneParameters(self, currentSceneParams):
        for key in currentSceneParams:
            self.sceneParams[key] = currentSceneParams[key]
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

    def stringToPoint3f(self, newValue) -> mi.Point3f:
        result = newValue.split(",")
        errMsg = (
            "Invalid XYZ input. Please use the following format: 1.0, 1.0, 1.0"
        )
        if len(result) != 3:
            raise ValueError(errMsg)
        try:
            x = float(result[0])
            y = float(result[1])
            z = float(result[2])
            return mi.Point3f(x, y, z)
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

    def setMinError(self, value: str):
        if not MaterialOptimizerModel.is_float(value):
            raise ValueError("Please provide a valid float value (e.g. 0.001)")
        self.minError = float(value)

    def setIterationCount(self, value: str):
        if not MaterialOptimizerModel.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 50)")
        self.iterationCount = int(value)

    def setSamplesPerPixel(self, value: str):
        if not MaterialOptimizerModel.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 4)")
        self.samplesPerPixel = int(value)

    def setLossFunction(self, value: str):
        self.lossFunction = value

    def setMarginPercentage(self, value: str):
        if not MaterialOptimizerModel.is_float(value):
            raise ValueError("Please provide a valid float value (e.g. 0.001)")

        floatValue = float(value)
        if value == INF_STR or floatValue >= 0.0:
            self.marginPercentage = floatValue
        else:
            raise ValueError(
                f"Valid values are: {INF_STR} or a positive floating points."
            )

    def setMarginPenalty(self, value: str):
        self.marginPenalty = value

    @staticmethod
    def minIdxInDrList(lis: list):
        minValue = dr.min(lis)
        return lis.index(minValue)

    def computeLoss(self, sensor: mi.Sensor, spp: int, seed: int = 0):
        # Perform a (noisy) differentiable rendering of the scene
        image = self.render(
            self.scene,
            sensor=sensor,
            spp=spp,
            params=self.sceneParams,
            seed=seed,
        )
        image = dr.clamp(image, 0.0, 1.0)

        # Evaluate the objective function from the current rendered image
        if self.lossFunction == MSE_STRING:
            result = self.mse(image, self.sensorToReferenceImageDict[sensor])
        elif self.lossFunction == BRIGHTNESS_IDP_MSE_STRING:
            result = self.brightnessIndependentMSE(
                image, self.sensorToReferenceImageDict[sensor]
            )
        elif self.lossFunction == DUAL_BUFFER_STRING:
            image2 = self.render(
                self.scene,
                sensor=sensor,
                spp=spp,
                params=self.sceneParams,
                seed=seed + 1,
            )
            image2 = dr.clamp(image2, 0.0, 1.0)
            result = self.dualBufferError(
                image, image2, self.sensorToReferenceImageDict[sensor]
            )
        elif self.lossFunction == MAE_STRING:
            result = self.mae(image, self.sensorToReferenceImageDict[sensor])
        elif self.lossFunction == MBE_STRING:
            result = self.mbe(image, self.sensorToReferenceImageDict[sensor])
        else:
            result = self.mse(image, self.sensorToReferenceImageDict[sensor])

        return result, image

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

    def dualBufferError(self, image, image2, refImage):
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
        return dr.mean((image - refImage) * (image2 - refImage))

    def mae(self, image, refImage):
        """L1 Loss: Mean Absolute Error"""
        return dr.mean(dr.abs(refImage - image))

    def mbe(self, image, refImage):
        """Mean Bias Error"""
        return dr.mean(refImage - image)

    def prepareOptimization(self, checkedRows: list):
        opts = self.initOptimizers(checkedRows)
        self.updateSceneParamsWithOptimizers(opts)
        sensorToInitImg = {
            sensor: self.render(self.scene, sensor, spp=256)
            for sensor in self.scene.sensors()
        }
        return opts, sensorToInitImg

    def optimizationLoop(
        self,
        opts: list,
        setProgressValue: callable = None,
        showDiffRender: callable = None,
    ):
        return self.optimizerStrategy.optimizationLoop(
            opts, setProgressValue, showDiffRender
        )

    def computeMargin(self, sensorLossOnPriorIt):
        if self.marginPercentage == float("inf"):
            return self.marginPercentage
        return sensorLossOnPriorIt * self.marginPercentage

    def penalizeLearningRates(self, opts, it):
        if self.marginPenalty == EXPONENTIAL_DECAY_STR:
            for opt in opts:
                newLearningRateDict = {
                    param: max(
                        0.0001,
                        self.exponentialDecay(
                            self.optimizationParams[param][
                                COLUMN_LABEL_LEARNING_RATE
                            ],
                            0.05,
                            it,
                        ),
                    )
                    for param in opt.variables.keys()
                }
                logging.info(f"New learning rates: {newLearningRateDict}")
                opt.set_learning_rate(newLearningRateDict)

    @staticmethod
    def exponentialDecay(original: float, decayFactor: float, time: int):
        """time = iteration in our case"""
        return original * (1 - decayFactor) ** time

    def updateLossAndSceneParamsHist(
        self, lossHist, sceneParamsHist, totalLoss
    ):
        sceneParamsHist.append(
            self.createSubsetSceneParams(
                self.sceneParams,
                SUPPORTED_MITSUBA_PARAMETER_PATTERNS,
            )
        )
        lossHist.append(totalLoss)

    def endOptimizationLog(self, sceneParamsHist, startTime, optLog):
        endTime = time.time()
        elapsedTime = endTime - startTime
        optLog.append(
            f"Optimization loop end. Elapsed time: {elapsedTime:.3f}s\n"
        )
        optLog.append(f"Initial scene parameters:\n {sceneParamsHist[0]}\n")
        optLog.append(f"End scene parameters:\n {sceneParamsHist[-1]}\n")
        optLog.append(f"Mitsuba version:\n {mi.__version__}\n")
        optLog.append("Hyperparameters:\n")
        optLog.append(f"\tminimum error:{self.minError},\n")
        optLog.append(f"\tspp:{self.samplesPerPixel},\n")
        optLog.append(f"\toptimization func.:{self.lossFunction},\n")
        optLog.append(f"\titeraration count:{self.iterationCount}\n")
        optLog.append(
            f"Optimization parameters:\n \t{self.optimizationParams}"
        )
        return "".join(optLog)

    def startOptimizationLog(self):
        startTime = time.time()
        optLog = ["Optimization loop begin\n"]
        return startTime, optLog

    def updateOptimizationLog(self, sceneParamsHist, optLog, it, totalLoss):
        currentItStr = f"Iteration {it:02d}"
        logging.info(currentItStr)
        optLog.append(currentItStr + "\n")
        currentTotalLossStr = f"\ttotal loss= {totalLoss:6f}"
        logging.info(currentTotalLossStr)
        optLog.append(currentTotalLossStr + "\n")
        optLog.append(f"Current scene parameters:\n \t{sceneParamsHist[it]}\n")

    def updateProgressBar(self, setProgressValue, itPercent):
        if setProgressValue is not None:
            setProgressValue(itPercent)

    def initPlotProgress(self, showDiffRender):
        if showDiffRender is not None:
            showDiffRender(
                diffRender=list(self.sensorToReferenceImageDict.values())[0],
                plotStatus=INITIAL_STATUS_STR,
            )

    def updatePlotProgress(
        self, showDiffRender, it, itPercent, diffRender, loss: float
    ):
        if showDiffRender is not None and (it == 1 or itPercent % 5 == 0):
            showDiffRender(
                self.convertToBitmap(diffRender), it, loss, RENDER_STATUS_STR
            )

    @staticmethod
    def convertToBitmap(image: mi.TensorXf):
        return mi.util.convert_to_bitmap(image)

    def setSensorToReferenceImageDict(self, readImages: list):
        # check length of reference images is equal to # of sensors
        sensors = self.scene.sensors()
        if len(readImages) != len(sensors):
            err = "The amount of selected reference images do not match "
            err += "with the available sensors loaded in the scene file."
            raise RuntimeError(err)

        # assuming the order of the loaded images corresponds to the appropriate
        # sensors defined in the loaded scene file
        self.sensorToReferenceImageDict = {
            sensors[idx]: readImg for idx, readImg in enumerate(readImages)
        }


class OptimizerStrategy(ABC):
    @abstractmethod
    def optimizationLoop(
        self,
        opts: list,
        setProgressValue: callable = None,
        showDiffRender: callable = None,
    ):
        pass


class DefaultOptimizerStrategy(OptimizerStrategy):
    def __init__(self, model: MaterialOptimizerModel) -> None:
        self.model = model

    def optimizationLoop(
        self,
        opts: list,
        setProgressValue: callable = None,
        showDiffRender: callable = None,
    ):
        lossHist = []
        sceneParamsHist = []
        sensors = self.model.scene.sensors()
        tmpLossTracker = {sensorIdx: [] for sensorIdx in range(len(sensors))}
        tmpFailTracker = 0

        startTime, optLog = self.model.startOptimizationLog()
        self.model.initPlotProgress(showDiffRender)
        for it in range(self.model.iterationCount):

            itPercent = int(it / self.model.iterationCount * 100)
            self.model.updateProgressBar(setProgressValue, itPercent)

            totalLoss = 0.0
            for sensorIdx, sensor in enumerate(sensors):
                currentLoss, diffRender = self.model.computeLoss(
                    sensor=sensor, spp=self.model.samplesPerPixel, seed=it
                )
                totalLoss += currentLoss[0]

                dr.backward(currentLoss)
                self.model.updateAfterStep(opts, self.model.sceneParams)
                if it > 0:
                    sensorLossOnPriorIt = tmpLossTracker[sensorIdx][-1]
                    margin = self.model.computeMargin(sensorLossOnPriorIt)
                    if currentLoss[0] > sensorLossOnPriorIt + margin:
                        msg = f"Current loss ({currentLoss[0]:.3f}) is larger"
                        msg += f" than loss on prior iteration plus some"
                        msg += (
                            f" margin ({(sensorLossOnPriorIt + margin):.3f})"
                        )
                        msg += f" on sensor index {sensorIdx}."
                        logging.info(msg)
                        tmpFailTracker += 1
                        if tmpFailTracker % 3 == 0:
                            self.model.penalizeLearningRates(opts, it)
                tmpLossTracker[sensorIdx].append(currentLoss[0])

            self.model.updatePlotProgress(
                showDiffRender, it, itPercent, diffRender, totalLoss
            )
            self.model.updateLossAndSceneParamsHist(
                lossHist, sceneParamsHist, totalLoss
            )
            self.model.updateOptimizationLog(
                sceneParamsHist, optLog, it, totalLoss
            )
            if totalLoss < self.model.minError:
                break

        if showDiffRender:
            showDiffRender(diffRender=None, plotStatus=CLOSE_STATUS_STR)
        optLog = self.model.endOptimizationLog(
            sceneParamsHist, startTime, optLog
        )

        return lossHist, sceneParamsHist, optLog


class CustomOptimizerStrategy(OptimizerStrategy):
    def __init__(self, model: MaterialOptimizerModel) -> None:
        self.model = model

    def optimizationLoop(
        self,
        opts: list,
        setProgressValue: callable = None,
        showDiffRender: callable = None,
    ):
        lossHist = []
        sceneParamsHist = []
        sensors = self.model.scene.sensors()
        tmpLossTracker = {sensorIdx: [] for sensorIdx in range(len(sensors))}
        tmpFailTracker = 0

        startTime, optLog = self.model.startOptimizationLog()
        self.model.initPlotProgress(showDiffRender)
        step_size = 16
        it = 0
        spp = self.model.samplesPerPixel

        while it < self.model.iterationCount:
            for step in range(step_size):

                itPercent = int(it / self.model.iterationCount * 100)
                self.model.updateProgressBar(setProgressValue, itPercent)

                totalLoss = 0.0
                for sensorIdx, sensor in enumerate(sensors):
                    currentLoss, diffRender = self.model.computeLoss(
                        sensor=sensor, spp=spp, seed=it
                    )
                    totalLoss += currentLoss[0]

                    dr.backward(currentLoss)
                    self.model.updateAfterStep(
                        opts, self.model.sceneParams, it
                    )
                    if it > 0:
                        sensorLossOnPriorIt = tmpLossTracker[sensorIdx][-1]
                        margin = self.model.computeMargin(sensorLossOnPriorIt)
                        if currentLoss[0] > sensorLossOnPriorIt + margin:
                            msg = f"Current loss ({currentLoss[0]}) is larger"
                            msg += f" than loss on prior iteration plus some"
                            msg += f" margin ({sensorLossOnPriorIt + margin})"
                            msg += f" on sensor index {sensorIdx}."
                            logging.info(msg)
                            tmpFailTracker += 1
                            if tmpFailTracker % 3 == 0:
                                self.model.penalizeLearningRates(opts, it)
                    tmpLossTracker[sensorIdx].append(currentLoss[0])

                self.model.updatePlotProgress(
                    showDiffRender, it, itPercent, diffRender, totalLoss
                )
                self.model.updateLossAndSceneParamsHist(
                    lossHist, sceneParamsHist, totalLoss
                )
                self.model.updateOptimizationLog(
                    sceneParamsHist, optLog, it, totalLoss
                )

                it += 1
                if (
                    it > self.model.iterationCount
                    or totalLoss < self.model.minError
                ):
                    if showDiffRender:
                        showDiffRender(
                            diffRender=None, plotStatus=CLOSE_STATUS_STR
                        )
                    optLog = self.model.endOptimizationLog(
                        sceneParamsHist, startTime, optLog
                    )
                    return lossHist, sceneParamsHist, optLog

            grid_res = min(
                128,
                self.model.sceneParams[GRID_VOLUME_DATA_SIGMA_T_STR].shape[0]
                * 2,
            )
            step_size = min(64, step_size * 2)
            spp = min(2, spp * 2)
            logging.info(
                f"New configuration: ic={step_size}, res={grid_res}, spp={spp}"
            )
            for opt in opts:
                opt[GRID_VOLUME_DATA_SIGMA_T_STR] = dr.upsample(
                    opt[GRID_VOLUME_DATA_SIGMA_T_STR],
                    shape=(grid_res, grid_res, grid_res),
                )
                self.model.sceneParams.update(opt)

        if showDiffRender:
            showDiffRender(diffRender=None, plotStatus=CLOSE_STATUS_STR)
        optLog = self.model.endOptimizationLog(
            sceneParamsHist, startTime, optLog
        )

        return lossHist, sceneParamsHist, optLog
