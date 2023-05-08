import logging
import time
from typing import Union

import mitsuba as mi
import drjit as dr
from src.constants import (
    ADVANCED_VERTEX_OPTIMIZATION_STRATEGY_LABEL,
    CLOSE_STATUS_STR,
    VERTEX_POSITIONS_PATTERN,
)
import src.material_optimizer_model as model
import torch


class AdvancedVertexOptimizer(model.OptimizerStrategy):
    """
    Implements a optimization strategy for shape reconstruction using advanced vertex position optimization.
    Highly influenced by:
     - Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob. 2021. Large Steps in
       Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
     - https://github.com/mitsuba-renderer/mitsuba3/discussions/600
    """

    def __init__(self, model: model.MaterialOptimizerModel) -> None:
        self.model = model
        self.label = ADVANCED_VERTEX_OPTIMIZATION_STRATEGY_LABEL

    def updateOptimizationLog(
        self, optLog, sensorIdx, currentLoss, regularizationLoss
    ) -> Union[list, list, str, dict]:
        lossStr = f"Sensor: {sensorIdx:02d}, "
        lossStr += f"Loss: {currentLoss[0]:.4f}"
        if dr.grad_enabled(regularizationLoss):
            lossStr += f" (regularizationLoss: {regularizationLoss[0]:.4f})"
        logging.info(lossStr)
        optLog.append(lossStr + "\n")

    @staticmethod
    def getParameterizationMatrix(
        positions: mi.Float, faces: mi.UInt, lambda_: float
    ):
        """
        Influenced from
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(len(positions) // 3, 3))
        faces = mi.TensorXi(faces, shape=(len(faces) // 3, 3))
        positions = positions.torch().cuda()
        faces = faces.torch().cuda()
        from largesteps.geometry import compute_matrix

        return compute_matrix(positions, faces, lambda_, alpha=0.5, cotan=True)

    @staticmethod
    def toDiff(M, positions):
        """
        Workaround for the kernel-crash issue mentioned in
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600#discussioncomment-5722753
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(len(positions) // 3, 3))
        positions = positions.torch().cuda()
        u = (M @ positions).cpu().numpy()
        u = mi.TensorXf(u.flatten(), shape=u.shape)
        return u.array

    @staticmethod
    def fromDiff(
        M: torch.Tensor, positions: mi.Float, method="Cholesky"
    ) -> mi.Float:
        """
        Influenced from
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(len(positions) // 3, 3))

        @dr.wrap_ad(source="drjit", target="torch")
        def from_diff_internal(up: torch.Tensor):
            from largesteps.parameterize import from_differential

            return from_differential(M, up, method)

        return from_diff_internal(positions).array

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

        # prepare for regularization loss
        vertexPosParamLabel, facesParamLabel = self.getParamLabels(opts)
        M = self.getParameterizationMatrix(
            self.model.sceneParams[vertexPosParamLabel],
            self.model.sceneParams[facesParamLabel],
            lambda_=19,
        )
        u = self.toDiff(M, self.model.sceneParams[vertexPosParamLabel])
        dr.set_grad_enabled(u, True)

        startTime, optLog = self.model.startOptimizationLog()
        self.model.initPlotProgress(showDiffRender)
        for it in range(self.model.iterationCount):

            itPercent = int(it / self.model.iterationCount * 100)
            self.model.updateProgressBar(setProgressValue, itPercent)

            totalLoss = 0.0
            for sensorIdx, sensor in enumerate(sensors):
                # image loss
                currentLoss, diffRender = self.model.computeLoss(
                    sensor=sensor, spp=self.model.samplesPerPixel, seed=seed
                )
                dr.backward(currentLoss)
                seed += 1 + sensorsSize
                diffRenderHist[sensorIdx].append(diffRender)

                # Evaluate regularization loss
                v = self.fromDiff(M, u)
                regLoss = dr.mean(
                    dr.abs(v - self.model.sceneParams[vertexPosParamLabel])
                )
                if dr.grad_enabled(regLoss):
                    dr.backward(regLoss)
                currentLoss += dr.detach(regLoss)
                totalLoss += currentLoss[0]

                self.updateOptimizationLog(
                    optLog, sensorIdx, currentLoss, regLoss
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
            if totalLoss < self.model.minError:
                break

        if showDiffRender:
            showDiffRender(diffRender=None, plotStatus=CLOSE_STATUS_STR)
        optLog = self.model.endOptimizationLog(
            sceneParamsHist, startTime, optLog
        )

        return lossHist, sceneParamsHist, optLog, diffRenderHist

    def getParamLabels(self, opts):
        vertexPosParamLabel = self.model.getParamLabel(
            VERTEX_POSITIONS_PATTERN, opts
        )
        if vertexPosParamLabel is None:
            raise RuntimeError(
                "Unexpected behavior during advanced vertex pos. optimization."
            )
        parentStr = vertexPosParamLabel.replace(".vertex_positions", "")
        facesParamLabel = parentStr + ".faces"
        return vertexPosParamLabel, facesParamLabel

    def checkOptimizationPreconditions(
        self, checkedRows: list
    ) -> Union[bool, str]:
        """
        Return True, if preconditions are fulfilled. Otherwise, return False
        and an error message.

        Precondition: User must select only one scene parameter label
        with '*.vertex_positions'.
        """
        if (
            self.model.countPatternInList(
                VERTEX_POSITIONS_PATTERN, checkedRows
            )
            == 1
        ):
            return True, ""

        msg = "For advanced vertex pos. optimization only one occurance of"
        msg += " '*.vertex_positions' is allowed in the selected"
        msg += " scene parameters."
        return False, msg
