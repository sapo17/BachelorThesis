import logging
import time
from typing import Union
from typing_extensions import override

import numpy as np
import mitsuba as mi
import drjit as dr
from src.constants import (
    CLOSE_STATUS_STR,
    GRID_VOLUME_OPTIMIZATION_STRATEGY_LABEL,
    SIGMA_T_PATTERN,
)
import src.material_optimizer_model as model


class GridVolumeOptimizer(model.OptimizerStrategy):
    """
    Implements a optimization strategy for shape reconstruction using grid
    volume optimization.
    """

    def __init__(self, model: model.MaterialOptimizerModel) -> None:
        self.model = model
        self.label = GRID_VOLUME_OPTIMIZATION_STRATEGY_LABEL

    def updateOptimizationLog(
        self,
        optLog,
        sensorIdx,
        currentLoss,
        regularizationLoss,
        gridVolumeParamLabel,
    ) -> Union[list, list, str, dict]:
        lossStr = f"Sensor: {sensorIdx:02d}, "
        lossStr += f"Loss: {currentLoss[0]:.4f}"
        if dr.grad_enabled(regularizationLoss):
            scaledLoss = (1e4 * regularizationLoss[0]) / dr.prod(
                self.model.sceneParams[gridVolumeParamLabel].shape
            )
            lossStr += f" (reg (avg. x 1e4): {scaledLoss:.4f})"
        logging.info(lossStr)
        optLog.append(lossStr + "\n")

    @override
    def optimizationLoop(
        self,
        opts: list,
        setProgressValue: callable = None,
        showDiffRender: callable = None,
    ):
        lossHist = []
        sceneParamsHist = []
        sensors = self.model.scene.sensors()
        sensorsSize = len(sensors)
        sensorWeight = 1 / sensorsSize
        diffRenderHist = {sensorIdx: [] for sensorIdx in range(sensorsSize)}
        tmpLossTracker = {sensorIdx: [] for sensorIdx in range(sensorsSize)}
        tmpFailTracker = 0

        startTime, optLog = self.model.startOptimizationLog()
        self.model.initPlotProgress(showDiffRender)
        step_size = 16
        it = 0
        spp = 1
        seed = 0
        gridVolumeParamLabel = self.model.getParamLabelFromOpts(
            SIGMA_T_PATTERN, opts
        )
        if gridVolumeParamLabel is None:
            raise RuntimeError(
                "Unexpected behavior during grid volume optimization."
            )

        while it < self.model.iterationCount:
            for _ in range(step_size):

                itPercent = int(it / self.model.iterationCount * 100)
                self.model.updateProgressBar(setProgressValue, itPercent)

                totalLoss = 0.0
                for sensorIdx, sensor in enumerate(sensors):
                    # Evaluate image loss
                    currentLoss, diffRender = self.model.computeLoss(
                        sensor=sensor, spp=spp, seed=seed
                    )
                    seed += 1 + sensorsSize
                    diffRenderHist[sensorIdx].append(diffRender)
                    dr.backward(currentLoss)

                    # Evaluate regularization loss
                    regularizationLoss = (
                        self.evalDiscreteLaplacianRegularazation(
                            self.model.sceneParams[gridVolumeParamLabel]
                        )
                    )
                    regularizationLoss *= sensorWeight
                    if dr.grad_enabled(regularizationLoss):
                        dr.backward(regularizationLoss)
                    currentLoss += dr.detach(regularizationLoss)
                    totalLoss += currentLoss[0]

                    self.updateOptimizationLog(
                        optLog,
                        sensorIdx,
                        currentLoss,
                        regularizationLoss,
                        gridVolumeParamLabel,
                    )
                    self.model.updateAfterStep(opts, self.model.sceneParams)
                    if it > 0:
                        sensorLossOnPriorIt = tmpLossTracker[sensorIdx][-1]
                        margin = self.model.computeMargin(sensorLossOnPriorIt)
                        if currentLoss[0] > sensorLossOnPriorIt + margin:
                            tmpFailTracker += 1
                            if tmpFailTracker % 3 == 0:
                                self.model.penalizeLearningRates(opts, it)
                    tmpLossTracker[sensorIdx].append(currentLoss[0])

                self.model.updateLossAndSceneParamsHist(
                    lossHist, sceneParamsHist, totalLoss
                )
                elapsedTime = time.time() - startTime
                self.model.updatePlotProgress(
                    showDiffRender,
                    it,
                    itPercent,
                    diffRenderHist[it % sensorsSize][-1],
                    totalLoss,
                    lossHist,
                    f"{elapsedTime:.3f}s",
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
                    return lossHist, sceneParamsHist, optLog, diffRenderHist

            grid_res = min(
                self.model.initialSceneParams[gridVolumeParamLabel].shape[0]
                * 2**4,
                self.model.sceneParams[gridVolumeParamLabel].shape[0] * 2,
            )
            step_size = min(32, step_size * 2)
            spp = min(self.model.samplesPerPixel, spp * 2)
            logging.info(
                f"New configuration: ic={step_size}, res={grid_res}, spp={spp}"
            )
            for opt in opts:
                if gridVolumeParamLabel in opt.variables:
                    opt[gridVolumeParamLabel] = dr.upsample(
                        opt[gridVolumeParamLabel],
                        shape=(grid_res, grid_res, grid_res),
                    )
                    self.model.sceneParams.update(opt)

        if showDiffRender:
            showDiffRender(diffRender=None, plotStatus=CLOSE_STATUS_STR)
        optLog = self.model.endOptimizationLog(
            sceneParamsHist, startTime, optLog
        )

        return lossHist, sceneParamsHist, optLog, diffRenderHist

    @override
    def checkOptimizationPreconditions(
        self, checkedRows: list
    ) -> Union[bool, str]:
        """
        Return True, if preconditions are fulfilled. Otherwise, return False
        and an error message.

        Precondition: User must select only one scene parameter label
        with '*.sigma_t.data'.
        """
        if self.model.countPatternInList(SIGMA_T_PATTERN, checkedRows) == 1:
            return True, ""

        msg = "For grid volume optimization only one occurance of"
        msg += " '*.sigma_t.data' is allowed in the selected scene parameters."
        return False, msg

    @override
    def output(self, paramLabel, paramValue, outputFileDir):
        """
        Implements an output strategy for given parameters.

        - paramLabel: scene parameter label
        - paramLabel: scene parameter value
        - outputFileDir: output directory for the output task
        """
        if SIGMA_T_PATTERN.search(paramLabel):

            # apply marching cubes
            from skimage import measure

            valNpy = np.array(paramValue)[:, :, :, 0]
            verts, faces, normals, values = measure.marching_cubes(
                valNpy,
                dr.min(paramValue)[0] + 0.2 * np.std(paramValue),
                allow_degenerate=False,
            )

            verts, faces = self.convert_obj_to_br(
                verts, faces, valNpy.shape[0]
            )
            outputObjName = (
                outputFileDir + f"//optimized_volume_{paramLabel}.obj"
            )
            self.marching_cubes_to_obj(
                (verts, faces, normals, values), outputObjName
            )

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

    @staticmethod
    def evalDiscreteLaplacianRegularazation(data, _=None):
        """
        Taken from Vicini et al. 2022, Differentiable SDF Rendering.
        Simple discrete laplacian regularization to encourage smooth surfaces.
        """

        def linerIdx(p):
            p.x = dr.clamp(p.x, 0, data.shape[0] - 1)
            p.y = dr.clamp(p.y, 0, data.shape[1] - 1)
            p.z = dr.clamp(p.z, 0, data.shape[2] - 1)
            return (
                p.z * data.shape[1] * data.shape[0] + p.y * data.shape[0] + p.x
            )

        shape = data.shape
        z, y, x = dr.meshgrid(
            *[dr.arange(mi.Float, shape[i]) for i in range(3)], indexing="ij"
        )
        p = mi.Point3i(x, y, z)
        c = dr.gather(mi.Float, data.array, linerIdx(p))
        vx0 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(-1, 0, 0))
        )
        vx1 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(1, 0, 0))
        )
        vy0 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(0, -1, 0))
        )
        vy1 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(0, 1, 0))
        )
        vz0 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(0, 0, -1))
        )
        vz1 = dr.gather(
            mi.Float, data.array, linerIdx(p + mi.Vector3i(0, 0, 1))
        )
        laplacian = dr.sqr(c - (vx0 + vx1 + vy0 + vy1 + vz0 + vz1) / 6)
        return dr.sum(laplacian)
