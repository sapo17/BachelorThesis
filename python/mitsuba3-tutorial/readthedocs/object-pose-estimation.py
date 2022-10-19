import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt


def createIntegrator(integratorType):
    integrator = {
        'type': integratorType,
    }
    return integrator


def apply_transformations(params, key, opt, optKey, transformationTo):
    opt[optKey] = dr.clamp(opt[optKey], -0.5, 0.5)

    trafo = mi.Transform4f.translate([opt[optKey].x, opt[optKey].y, 0.0])

    params[key] = dr.ravel(trafo @ transformationTo)
    params.update()


mi.set_variant('cuda_ad_rgb')
scene = mi.load_file(
    'python/mitsuba3-tutorial/scenes/editing-a-scene/cbox.xml')

img_ref = mi.render(scene, spp=1024)
img_to_show = mi.util.convert_to_bitmap(img_ref)
plt.title('Reference')
plt.imshow(img_to_show)
plt.show()

params = mi.traverse(scene)
sceneKey = 'light.vertex_positions'
initial_vertex_positions = dr.unravel(mi.Point3f, params[sceneKey])

opt = mi.ad.Adam(lr=0.01)
optKey = 'trans'
opt[optKey] = mi.Point2f(0.1, -0.25)

# initial state
apply_transformations(params, sceneKey, opt, optKey,
                      transformationTo=initial_vertex_positions)

img_init = mi.render(scene, seed=0, spp=1024)
img_to_show = mi.util.convert_to_bitmap(img_init)
plt.title('Initial image')
plt.imshow(img_to_show)
plt.show()


iteration_count = 30
spp = 16

loss_hist = []
for it in range(iteration_count):
    # apply mesh transformation
    apply_transformations(params, sceneKey, opt, optKey,
                          transformationTo=initial_vertex_positions)

    # perform differentiable rendering
    img = mi.render(scene, params, seed=it, spp=spp)

    # evaluate the objective function, gx, L2
    loss = dr.sum(dr.sqr(img - img_ref)) / len(img)

    # backpropagate through the rendering process
    dr.backward(loss)

    # optimizer: take a GDS
    opt.step()

    loss_hist.append(loss)
    print(
        f"Iteration {it:02d}: error={loss[0]:6f}, trans=[{opt['trans'].x[0]:.4f}, {opt['trans'].y[0]:.4f}]", end='\r')


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(loss_hist)
axs[0][0].set_xlabel('iteration')
axs[0][0].set_ylabel('Loss')
axs[0][0].set_title('Parameter error plot')

axs[0][1].imshow(mi.util.convert_to_bitmap(img_init))
axs[0][1].axis('off')
axs[0][1].set_title('Initial Image')

axs[1][0].imshow(mi.util.convert_to_bitmap(mi.render(scene, spp=1024)))
axs[1][0].axis('off')
axs[1][0].set_title('Optimized image')

axs[1][1].imshow(mi.util.convert_to_bitmap(img_ref))
axs[1][1].axis('off')
axs[1][1].set_title('Reference Image')
plt.show()
