import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import matplotlib.pyplot as plt


def createIntegrator(integratorType):
    return {
        'type': integratorType
    }


mi.set_variant('cuda_ad_rgb')

integrator = createIntegrator('direct_reparam')
scene = mi.load_file(
    'python\mitsuba3-tutorial\scenes\editing-a-scene\cbox.xml')

img_ref = mi.render(scene, spp=1024)
img_converted = mi.util.convert_to_bitmap(img_ref)
plt.imshow(img_converted)
plt.title('Reference')
plt.show()

params = mi.traverse(scene)
key = 'light.vertex_positions'
light_init_vertex_pos = dr.unravel(mi.Point3f, params[key])


# only optimize the translation param
opt = mi.ad.Adam(lr=0.025)
optKey = 'trans'
opt[optKey] = mi.Point2f(0, -0.5)

# From the optimizer’s point of view, those variables are the same as any other
# variables optimized in the previous tutorials, to the exception that when
# calling opt.update(), the optimizer doesn’t know how to propagate their new
# values to the scene parameters. This has to be done manually, and we
# encapsulate exactly that logic in the function defined below.
# After clamping the optimized variables to a proper range, this function
# creates a transformation object combining a translation and rotation and
# applies it to the vertex positions stored previously. It then flattens those
# new vertex positions before assigning them to the scene parameters.


def apply_transformations(params, light_init_vertex_pos, opt, optKey, paramKey):
    opt[optKey] = dr.clamp(opt[optKey], -0.5, 0.5)

    trafo = mi.Transform4f.translate([opt[optKey].x, opt[optKey].y, 0.0])

    params[paramKey] = dr.ravel(
        trafo @ light_init_vertex_pos)
    params.update()


# initial state
apply_transformations(params, light_init_vertex_pos, opt, optKey, key)

img_init = mi.render(scene, seed=0, spp=1024)

img_converted = mi.util.convert_to_bitmap(img_init)
plt.imshow(img_converted)
plt.title('Image Init')
plt.show()
