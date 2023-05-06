import time
from src.constants import CLOSE_STATUS_STR
import drjit as dr
import src.material_optimizer_model as model


class DefaultOptimizerStrategy(model.OptimizerStrategy):
    """
    Implements the original (and currently default) optimization strategy.
    """

    def __init__(self, model: model.MaterialOptimizerModel) -> None:
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
        sensorsSize = len(sensors)
        diffRenderHist = {sensorIdx: [] for sensorIdx in range(sensorsSize)}
        tmpLossTracker = {sensorIdx: [] for sensorIdx in range(sensorsSize)}
        tmpFailTracker = 0
        seed = 0

        startTime, optLog = self.model.startOptimizationLog()
        self.model.initPlotProgress(showDiffRender)
        for it in range(self.model.iterationCount):

            itPercent = int(it / self.model.iterationCount * 100)
            self.model.updateProgressBar(setProgressValue, itPercent)

            totalLoss = 0.0
            for sensorIdx, sensor in enumerate(sensors):
                currentLoss, diffRender = self.model.computeLoss(
                    sensor=sensor, spp=self.model.samplesPerPixel, seed=seed
                )
                seed += 1 + sensorsSize
                diffRenderHist[sensorIdx].append(diffRender)
                totalLoss += currentLoss[0]

                dr.backward(currentLoss)
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
            if totalLoss < self.model.minError:
                break

        if showDiffRender:
            showDiffRender(diffRender=None, plotStatus=CLOSE_STATUS_STR)
        optLog = self.model.endOptimizationLog(
            sceneParamsHist, startTime, optLog
        )

        return lossHist, sceneParamsHist, optLog, diffRenderHist
