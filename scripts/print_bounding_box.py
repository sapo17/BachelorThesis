import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

def getShape(scene: mi.Scene, id: str):
    for shape in scene.shapes():
        if shape.id() == id:
            return shape
    return None

scene = mi.load_file('scenes\material-preview\\bunny-multi-sensor\multi-cam-object-extended.xml', spp=1)
shape = getShape(scene, "bunny")

bbox = shape.bbox()
bsphere = bbox.bounding_sphere()
print(bbox)
print(bsphere)

# birdy
# Min: -0.875, -0.933, 9.320
# Max: 0.586, 0.723, 10.944

# green alginate
# Min: -0.235536, 0.032117, -0.283350 min x can go lower -0.236?, min z can go lower -0.285? e.g. -0.236, 0.032117, -0.28350
# Max: -0.204037, 0.032658, -0.254136 max x can go higher -0.2039?, max z can go more higher -0.251? e.g. -0.20395, 0.032658, -0.253

# multi sensor bunny
# BoundingBox3f[
#   min = [-2.22219, 0.0573322, -1.81878],
#   max = [2.35461, 4.59399, 1.72843]
# ]
# BoundingSphere3f[
#   center = [0.0662107, 2.32566, -0.0451787],
#   radius = 3.67801
# ]