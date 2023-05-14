import logging
import time
from typing import Union
from typing_extensions import override

import mitsuba as mi
import drjit as dr
import numpy as np
from src.constants import (
    ADVANCED_VERTEX_OPTIMIZATION_STRATEGY_LABEL,
    BASE_COLOR_DATA_PATTERN,
    CLOSE_STATUS_STR,
    COLUMN_LABEL_BETA_1,
    COLUMN_LABEL_BETA_2,
    COLUMN_LABEL_LEARNING_RATE,
    COLUMN_LABEL_MAX_CLAMP_LABEL,
    COLUMN_LABEL_MIN_CLAMP_LABEL,
    DEFAULT_LAMBDA_PARAM_MATRIX_VALUE,
    FALSE_TRUE_IN_STR,
    NONE_STR,
    VERTEX_COLOR_PATTERN,
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
        self.toggleUseUniformAdam(FALSE_TRUE_IN_STR[0])
        self.setLambda(DEFAULT_LAMBDA_PARAM_MATRIX_VALUE)
        self.setRemeshStepSize(NONE_STR)

    def toggleUseUniformAdam(self, text):
        if text == FALSE_TRUE_IN_STR[1]:
            self.useUniformAdam = True
        else:
            self.useUniformAdam = False

    def setLambda(self, value: str):
        if not self.model.is_int(value):
            raise ValueError("Please provide a valid integer value.")
        intVal = int(value)
        if 1 <= intVal <= 99:
            self.lambda_ = intVal
        else:
            raise ValueError(
                "Please provide a valid integer value in [1, 99]."
            )

    def setRemeshStepSize(self, value):
        if value == "None":
            self.remeshStepSize = None
            return

        if not self.model.is_int(value):
            raise ValueError("Please provide a valid integer value (e.g. 32).")

        self.remeshStepSize = int(value)

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
        positions: mi.Float, posLen: int, faces: mi.UInt, lambda_: float
    ):
        """
        Influenced from
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(posLen // 3, 3))
        faces = mi.TensorXi(faces, shape=(len(faces) // 3, 3))
        positions = positions.torch().cuda()
        faces = faces.torch().cuda()
        from largesteps.geometry import compute_matrix

        return compute_matrix(positions, faces, lambda_, cotan=True)

    @staticmethod
    def toDiff(M, positions: mi.Float, posLen: int):
        """
        Workaround for the kernel-crash issue mentioned in
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600#discussioncomment-5722753
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(posLen // 3, 3))
        positions = positions.torch().cuda()
        u = (M @ positions).cpu().numpy()
        u = mi.TensorXf(u.flatten(), shape=u.shape)
        return u.array

    @staticmethod
    def fromDiff(
        M: torch.Tensor, positions: mi.Float, posLen: int, method="Cholesky"
    ) -> mi.Float:
        """
        Influenced from
        https://github.com/mitsuba-renderer/mitsuba3/discussions/600
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        positions = mi.TensorXf(positions, shape=(posLen // 3, 3))

        @dr.wrap_ad(source="drjit", target="torch")
        def from_diff_internal(up: torch.Tensor):
            from largesteps.parameterize import from_differential

            return from_differential(M, up, method)

        return from_diff_internal(positions).array

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
                # regularization: using the approach by Nicolet et al. 2021
                self.transformVertexPos()

                # image loss
                currentLoss, diffRender = self.model.computeLoss(
                    sensor=sensor, spp=self.model.samplesPerPixel, seed=seed
                )
                dr.backward(currentLoss)

                self.uOpt.step()
                self.model.updateAfterStep(opts, self.model.sceneParams)

                # penalize learning rates, if specified
                if it > 0:
                    sensorLossOnPriorIt = tmpLossTracker[sensorIdx][-1]
                    margin = self.model.computeMargin(sensorLossOnPriorIt)
                    if currentLoss[0] > sensorLossOnPriorIt + margin:
                        tmpFailTracker += 1
                        if tmpFailTracker % 3 == 0:
                            self.model.penalizeLearningRates(opts, it)
                            self.penalizeUOpt(it)

                # manage loop
                seed += 1 + sensorsSize
                diffRenderHist[sensorIdx].append(diffRender)
                totalLoss += currentLoss[0]
                tmpLossTracker[sensorIdx].append(currentLoss[0])

            # remesh, if specified
            if self.remeshStepSize != None:
                if it > 0 and it % self.remeshStepSize == 0:
                    self.remesh(opts, 5)

            # manage loop
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

    def penalizeUOpt(self, it):
        penalizedLr = self.model.exponentialDecay(
            self.model.optimizationParams[self.vertexPosParamLabel][
                COLUMN_LABEL_LEARNING_RATE
            ],
            0.05,
            it,
        )
        newLr = max(0.00001, penalizedLr)
        self.uOpt.set_learning_rate({self.vertexPosParamLabel: newLr})

    def transformVertexPos(self):
        self.u = self.uOpt[self.vertexPosParamLabel]
        v = self.fromDiff(self.M, self.u, len(self.u))
        self.model.sceneParams[self.vertexPosParamLabel] = v
        self.ensureLegalVertexPositions()
        self.model.sceneParams.update()

    def remesh(self, opts, iterations=10):
        (
            vertexPosParamLabel,
            facesParamLabel,
            parentStr,
        ) = self.getVertexAndFacesAndParentLabels(self.vertexPosParamLabel)
        isVpGradEnabled = dr.grad_enabled(
            self.model.sceneParams[vertexPosParamLabel]
        )

        # disable grad tracking
        if isVpGradEnabled:
            dr.set_grad_enabled(
                self.model.sceneParams[vertexPosParamLabel], False
            )

        # prepare vertices and faces
        vp = mi.TensorXf(
            self.model.sceneParams[vertexPosParamLabel],
            shape=(len(self.model.sceneParams[vertexPosParamLabel]) // 3, 3),
        )
        f = mi.TensorXi(
            self.model.sceneParams[facesParamLabel],
            shape=(len(self.model.sceneParams[facesParamLabel]) // 3, 3),
        )
        v_src = vp.torch().cuda()
        f_src = f.torch().cuda()
        v_unique, f_unique = self.remove_duplicates(v_src, f_src)
        v_cpu = v_unique.cpu().numpy()
        f_cpu = f_unique.cpu().numpy()

        # Target edge length
        h = (self.average_edge_length(v_unique, f_unique)).cpu().numpy() * 0.5

        from gpytoolbox import remesh_botsch

        # Run given iterations of the Botsch-Kobbelt remeshing algorithm
        v_new, f_new = remesh_botsch(
            v_cpu.astype(np.double),
            f_cpu.astype(np.int32),
            iterations,
            h,
            True,
        )

        # update object with the new vertices and faces
        v_src = torch.from_numpy(v_new).cuda().float().contiguous()
        f_src = torch.from_numpy(f_new).cuda().contiguous()
        v_unique, f_unique = self.remove_duplicates(v_src, f_src)
        vp = mi.TensorXf(v_src)
        f = mi.TensorXi(f_src)
        newVpCount = len(vp) // 3
        newFCount = len(f) // 3

        self.uOpt.reset(vertexPosParamLabel)
        self.model.sceneParams[parentStr + ".vertex_count"] = newVpCount
        self.model.sceneParams[vertexPosParamLabel] = vp.array
        self.model.sceneParams[parentStr + ".face_count"] = newFCount
        self.model.sceneParams[facesParamLabel] = f.array
        self.model.sceneParams.update()

        # update vertex colors if included in the optimization
        self.updateVertexColors(opts, newVpCount)

        # recalculate M and apply to u
        positions = self.model.sceneParams[self.vertexPosParamLabel]
        self.prepareParameterizationMatrix(
            np.array(positions),
            len(positions),
            self.model.sceneParams[facesParamLabel],
        )
        self.uOpt[self.vertexPosParamLabel] = self.u

        # (re)enable grad tracking
        if isVpGradEnabled:
            dr.set_grad_enabled(
                self.model.sceneParams[vertexPosParamLabel], True
            )
        logMsg = "Remeshed object."
        logMsg += f"New vertex and face count: ({newVpCount}, {newFCount})."
        logging.info(logMsg)

    def updateVertexColors(self, opts, newVpCount):
        vertexColorLabel, opt = self.model.getParamLabelAndOptFromOpts(
            VERTEX_COLOR_PATTERN, opts
        )
        if vertexColorLabel != None and dr.grad_enabled(
            self.model.sceneParams[vertexColorLabel]
        ):
            dr.set_grad_enabled(
                self.model.sceneParams[vertexColorLabel], False
            )
            self.model.sceneParams[vertexColorLabel] = np.zeros(3 * newVpCount)
            self.model.sceneParams.update()
            opt[vertexColorLabel] = self.model.sceneParams[vertexColorLabel]
            dr.set_grad_enabled(self.model.sceneParams[vertexColorLabel], True)

    def getParamLabelsFromOpts(self, opts):
        vertexPosParamLabel = self.model.getParamLabelFromOpts(
            VERTEX_POSITIONS_PATTERN, opts
        )
        if vertexPosParamLabel is None:
            raise RuntimeError(
                "Unexpected behavior during advanced vertex pos. optimization."
            )
        return self.getVertexAndFacesAndParentLabels(vertexPosParamLabel)

    def getParamLabelsFromParams(self, params):
        vertexPosParamLabel = None
        for param in params:
            if VERTEX_POSITIONS_PATTERN.match(param):
                vertexPosParamLabel = param
        if vertexPosParamLabel is None:
            raise RuntimeError(
                "Unexpected behavior during advanced vertex pos. optimization."
            )
        return self.getVertexAndFacesAndParentLabels(vertexPosParamLabel)

    def getVertexAndFacesAndParentLabels(self, vertexPosParamLabel):
        parentStr = vertexPosParamLabel.replace(".vertex_positions", "")
        facesParamLabel = parentStr + ".faces"
        return vertexPosParamLabel, facesParamLabel, parentStr

    @override
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
            if (
                self.model.countPatternInList(
                    BASE_COLOR_DATA_PATTERN, checkedRows
                )
                == 1
            ):
                msg = "Currently, texture and vertex pos. optimization isn't "
                msg += "supported. Please unselect either one to continue."
                msg += "Alternatively, vertex color optimization can be used "
                msg += "along with vertex pos. optimization."
                return False, msg

            return True, ""

        msg = "For advanced vertex pos. optimization only one occurance of"
        msg += " '*.vertex_positions' is allowed in the selected"
        msg += " scene parameters."
        return False, msg

    @staticmethod
    def remove_duplicates(v, f):
        """
        Generate a mesh representation with no duplicates and
        return it along with the mapping to the original mesh layout.

        Taken from
        https://github.com/rgl-epfl/large-steps-pytorch
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """

        unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
        new_faces = inverse[f.long()]
        return unique_verts, new_faces

    @staticmethod
    def average_edge_length(verts, faces):
        """
        Compute the average length of all edges in a given mesh.

        Parameters
        ----------
        verts : torch.Tensor
            Vertex positions.
        faces : torch.Tensor
            array of triangle faces.

        Taken from
        https://github.com/rgl-epfl/large-steps-pytorch
        Proposed approach by: Baptiste Nicolet, Alec Jacobson, and Wenzel Jakob.
        2021. Large Steps in Inverse Rendering of Geometry. ACM SIGGRAPH 40(6).
        """
        face_verts = verts[faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Side lengths of each triangle, of shape (sum(F_n),)
        # A is the side opposite v1, B is opposite v2, and C is opposite v3
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        return (A + B + C).sum() / faces.shape[0] / 3

    @override
    def initOptimizers(self, params: list) -> list:
        """
        Override of initOptimizers(). The chosen vertex position parameter will
        be handled differently. It won't be part of default list of optimizers
        (aka opts). However, the initialization of the corresponding ADAM
        optimizer is almost identical to the original implementation. Beware
        that other (selected) parameter(s) are initialized as in the original
        implementation.
        """

        # prepare parameterization matrix
        (
            self.vertexPosParamLabel,
            facesParamLabel,
            _,
        ) = self.getParamLabelsFromParams(params)
        positions = self.model.sceneParams[self.vertexPosParamLabel]
        self.prepareParameterizationMatrix(
            positions, len(positions), self.model.sceneParams[facesParamLabel]
        )

        opts = []
        for k in params:
            # vertex position: initialized similarly, but not part of "opts"
            if k == self.vertexPosParamLabel:
                self.uOpt = mi.ad.Adam(
                    lr=self.model.optimizationParams[k][
                        COLUMN_LABEL_LEARNING_RATE
                    ],
                    beta_1=self.model.optimizationParams[k][
                        COLUMN_LABEL_BETA_1
                    ],
                    beta_2=self.model.optimizationParams[k][
                        COLUMN_LABEL_BETA_2
                    ],
                    params={k: self.u},
                    mask_updates=True,
                    uniform=self.useUniformAdam,
                )
            else:
                # other parameters: same as original implementation
                opts.append(
                    mi.ad.Adam(
                        lr=self.model.optimizationParams[k][
                            COLUMN_LABEL_LEARNING_RATE
                        ],
                        beta_1=self.model.optimizationParams[k][
                            COLUMN_LABEL_BETA_1
                        ],
                        beta_2=self.model.optimizationParams[k][
                            COLUMN_LABEL_BETA_2
                        ],
                        params={k: self.model.sceneParams[k]},
                        mask_updates=True,
                    )
                )
        return opts

    def prepareParameterizationMatrix(self, positions, posLen: int, faces):
        self.M = self.getParameterizationMatrix(
            positions, posLen, faces, lambda_=self.lambda_
        )
        self.u = self.toDiff(self.M, positions, posLen)

    def ensureLegalVertexPositions(self):
        point = dr.unravel(
            mi.Point3f, self.model.sceneParams[self.vertexPosParamLabel]
        )
        minVal = self.model.optimizationParams[self.vertexPosParamLabel][
            COLUMN_LABEL_MIN_CLAMP_LABEL
        ]
        maxVal = self.model.optimizationParams[self.vertexPosParamLabel][
            COLUMN_LABEL_MAX_CLAMP_LABEL
        ]
        clampedPoint = dr.clamp(point, minVal, maxVal)
        self.model.sceneParams[self.vertexPosParamLabel] = dr.ravel(
            clampedPoint
        )

    @override
    def prepareOptimization(self, checkedRows: list):
        opts, sensorToInitImg = super().prepareOptimization(checkedRows)
        torch.cuda.empty_cache()
        return opts, sensorToInitImg

