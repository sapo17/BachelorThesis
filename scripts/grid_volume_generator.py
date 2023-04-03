import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

grid_res = 4

mi.VolumeGrid(
    dr.full(mi.TensorXf, 0.002, (grid_res, grid_res, grid_res, 1))
).write("empty.vol")
