"""
Author: Can Hasbay
"""

import os
import mitsuba as mi
import drjit as dr
import logging
from src.constants import *

# set mitsuba variant: NVIDIA CUDA
mi.set_variant(CUDA_AD_RGB)


class MaterialOptimizerModel:
    def __init__(self) -> None:
        self.refImage = None
        self.sceneRes = (256, 256)
        self.loadMitsubaScene()
        self.setSceneParams(self.scene)
        self.setInitialSceneParams(self.sceneParams)
        self.setDefaultOptimizationParams(self.initialSceneParams)
        self.setSamplesPerPixel(SUPPORTED_SPP_VALUES[0])
        self.setLossFunction(LOSS_FUNCTION_STRINGS[0])

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
                COLUMN_LABEL_LEARNING_RATE: 0.03,
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
        elif K_PATTERN.search(key):
            opt[key] = dr.clamp(opt[key], DEFAULT_MIN_CLAMP_VALUE, MAX_K_VALUE)
        elif ALPHA_PATTERN.search(key):
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

    def initOptimizers(self, params: list) -> list:
        opt = mi.ad.Adam(
            lr=0.2, params={k: self.sceneParams[k] for k in params}
        )
        opt.set_learning_rate(
            {
                k: self.optimizationParams[k][COLUMN_LABEL_LEARNING_RATE]
                for k in params
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

    @staticmethod
    def minIdxInDrList(lis: list):
        minValue = dr.min(lis)
        return lis.index(minValue)

    def computeLoss(self, seed: int = 0):
        # Perform a (noisy) differentiable rendering of the scene
        image = self.render(
            self.scene,
            spp=self.samplesPerPixel,
            params=self.sceneParams,
            seed=seed,
        )
        image = dr.clamp(image, 0.0, 1.0)

        # Evaluate the objective function from the current rendered image
        if self.lossFunction == MSE_STRING:
            result = self.mse(image, self.refImage)
        elif self.lossFunction == BRIGHTNESS_IDP_MSE_STRING:
            result = self.brightnessIndependentMSE(image, self.refImage)
        elif self.lossFunction == DUAL_BUFFER_STRING:
            image2 = self.render(
                self.scene,
                spp=self.samplesPerPixel,
                params=self.sceneParams,
                seed=seed + 1,
            )
            image2 = dr.clamp(image2, 0.0, 1.0)
            result = self.dualBufferError(image, image2)
        elif self.lossFunction == MAE_STRING:
            result = self.mae(image, self.refImage)
        elif self.lossFunction == MBE_STRING:
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

    def prepareOptimization(self, checkedRows: list):
        opts = self.initOptimizers(checkedRows)
        self.updateSceneParamsWithOptimizers(opts)
        initImg = self.render(self.scene, spp=256)
        return opts, initImg

    def optimizationLoop(self, opts: list, setProgressValue: callable = None):
        lossHist = []
        sceneParamsHist = []
        for it in range(self.iterationCount):

            if setProgressValue is not None:
                setProgressValue(int(it / self.iterationCount * 100))

            loss = self.computeLoss(it)
            lossHist.append(loss)
            sceneParamsHist.append(
                self.createSubsetSceneParams(
                    self.sceneParams,
                    SUPPORTED_MITSUBA_PARAMETER_PATTERNS,
                )
            )
            logging.info(f"Iteration {it:02d}")
            logging.info(f"\tcurrent loss= {loss[0]:6f}")

            if loss[0] < self.minError:
                break

            # Backpropagate through the rendering process
            dr.backward(loss)

            self.updateAfterStep(opts, self.sceneParams)

        return lossHist, sceneParamsHist
