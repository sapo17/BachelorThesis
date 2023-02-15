import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

def printMinMax(vertices: mi.Float):
    x = dr.unravel(mi.Point3f, vertices)[0]
    y = dr.unravel(mi.Point3f, vertices)[1]
    z = dr.unravel(mi.Point3f, vertices)[2]
    print(f"Min: {dr.min(x)[0]:.3f}, {dr.min(y)[0]:.3f}, {dr.min(z)[0]:.3f}")
    print(f"Max: {dr.max(x)[0]:.3f}, {dr.max(y)[0]:.3f}, {dr.max(z)[0]:.3f}")

scene = mi.load_file('scenes\metashape\mini-birdy-statue\multiple_sensors\scene.xml', spp=1)
params = mi.traverse(scene)
printMinMax(params['PLYMesh.vertex_positions'])
