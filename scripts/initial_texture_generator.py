import mitsuba as mi
import drjit as dr

mi.set_variant("cuda_ad_rgb")

init_texture = dr.zeros(mi.TensorXf, (512, 512, 3))
mi.util.write_bitmap('init-texture-512.png', init_texture)