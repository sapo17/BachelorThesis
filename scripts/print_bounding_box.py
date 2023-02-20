import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

def printMinMax(vertices: mi.Float):
    x = dr.unravel(mi.Point3f, vertices)[0]
    y = dr.unravel(mi.Point3f, vertices)[1]
    z = dr.unravel(mi.Point3f, vertices)[2]
    print(f"Min: {dr.min(x)[0]:.6f}, {dr.min(y)[0]:.6f}, {dr.min(z)[0]:.6f}")
    print(f"Max: {dr.max(x)[0]:.6f}, {dr.max(y)[0]:.6f}, {dr.max(z)[0]:.6f}")

scene = mi.load_file('scenes\\texture-scanner\samples-231122\greenish-circular-obj\scene-first-rgb.xml', spp=1)
# scene = mi.load_file('scenes\metashape\mini-birdy-statue\multiple_sensors\scene.xml', spp=1)
params = mi.traverse(scene)
printMinMax(params['material-ply.vertex_positions'])

# birdy
# Min: -0.875, -0.933, 9.320
# Max: 0.586, 0.723, 10.944

# green alginate
# Min: -0.235536, 0.032117, -0.283350 min x can go lower -0.236?, min z can go lower -0.285? e.g. -0.236, 0.032117, -0.28350
# Max: -0.204037, 0.032658, -0.254136 max x can go higher -0.2039?, max z can go more higher -0.251? e.g. -0.20395, 0.032658, -0.253