import logging
import time
from src.constants import CLOSE_STATUS_STR, GRID_VOLUME_DATA_SIGMA_T_STR
import drjit as dr
import src.material_optimizer_model as model


class GridVolumeOptimizer(model.OptimizerStrategy):
    """
    Implements a optimization strategy for shape reconstruction using grid
    volume optimization.
    """

    def __init__(self, model: model.MaterialOptimizerModel) -> None:
        self.model = model

    def updateOptimizationLog(
        self, optLog, sensorIdx, currentLoss, regularizationLoss
    ):
        lossStr = f"Sensor: {sensorIdx:02d}, "
        lossStr += f"Loss: {currentLoss[0]:.4f}"
        if dr.grad_enabled(regularizationLoss):
            scaledLoss = (1e4 * regularizationLoss[0]) / dr.prod(
                self.model.sceneParams[GRID_VOLUME_DATA_SIGMA_T_STR].shape
            )
            lossStr += f" (reg (avg. x 1e4): {scaledLoss:.4f})"
        logging.info(lossStr)
        optLog.append(lossStr + "\n")

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
                        self.model.evalDiscreteLaplacianRegularazation(
                            self.model.sceneParams[
                                GRID_VOLUME_DATA_SIGMA_T_STR
                            ]
                        )
                    )
                    regularizationLoss *= sensorWeight
                    if dr.grad_enabled(regularizationLoss):
                        dr.backward(regularizationLoss)
                    currentLoss += dr.detach(regularizationLoss)
                    totalLoss += currentLoss[0]

                    self.updateOptimizationLog(
                        optLog, sensorIdx, currentLoss, regularizationLoss
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
                self.model.initialSceneParams[
                    GRID_VOLUME_DATA_SIGMA_T_STR
                ].shape[0]
                * 2**4,
                self.model.sceneParams[GRID_VOLUME_DATA_SIGMA_T_STR].shape[0]
                * 2,
            )
            step_size = min(32, step_size * 2)
            spp = min(self.model.samplesPerPixel, spp * 2)
            logging.info(
                f"New configuration: ic={step_size}, res={grid_res}, spp={spp}"
            )
            for opt in opts:
                if GRID_VOLUME_DATA_SIGMA_T_STR in opt.variables:
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

        return lossHist, sceneParamsHist, optLog, diffRenderHist
