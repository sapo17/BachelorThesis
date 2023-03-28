import mitsuba as mi
import drjit as dr
import numpy as np

mi.set_variant("cuda_ad_rgb")

# 0.14734427630901337, 0.22982807457447052, 0.18363985419273376
r = np.full(fill_value=0.14734427630901337, shape=(512,512) )
g = np.full(fill_value=0.22982807457447052, shape=(512,512) )
b = np.full(fill_value=0.18363985419273376, shape=(512,512) )
rgb = np.dstack((r,g,b))

# init_texture = dr.zeros(mi.TensorXf, (512, 512, 3))
init_texture = mi.TensorXf(rgb)
mi.util.write_bitmap('init-texture-blue-it-14-res-512.png', init_texture)