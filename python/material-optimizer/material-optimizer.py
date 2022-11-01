
from matplotlib import pyplot as plt
import drjit as dr
import mitsuba as mi
import warnings
import re
import logging
from pathlib import Path


log_file = Path("python\material-optimizer\material-optimizer.log")
log_file.unlink(missing_ok=True)
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

mi.set_variant('cuda_ad_rgb')


def create_subset_scene_params(params: mi.SceneParameters, pattern: re.Pattern):
    return {k: mi.Color3f(v) for k, v in params.items() if pattern.search(k)}


def randomize_srgb_params(params: mi.SceneParameters, ref_params: dict, bound: float):
    if bound < 0.0:
        warnings.warn('Negative bound value! Using 1.0 as the bound value.')
        bound = 1.0

    num_of_channels = 3
    rng = mi.PCG32(size=num_of_channels * len(ref_params))
    samples = rng.next_float64() * bound

    for i, key in enumerate(ref_params):
        if type(params[key]) is not mi.Color3f:
            warn_msg = 'Invalid type:' + str(type(params[key]))
            warnings.warn(warn_msg)
            raise ValueError(
                'Given ref_params dictionary values must have the mi.Color3f type!')
        params[key] = mi.Color3f(
            samples[i*num_of_channels], samples[i*num_of_channels+1], samples[i*num_of_channels+2])

    params.update()


# load scene and get scene params
scene = mi.load_file(
    'python/mitsuba3-tutorial/scenes/cbox-sch-modified/cbox-multiple-lights-2.xml', res=256, integrator='prb')
params = mi.traverse(scene)


# construct a dict from scene params that contains radiance.value
radiance_pattern = re.compile(r'.*\.radiance\.value')
ref_radiance_params = create_subset_scene_params(params, radiance_pattern)

# construct a dict from scene params that contains reflectance.value
reflectance_pattern = re.compile(r'.*\.reflectance\.value')
ref_reflectance_params = create_subset_scene_params(
    params, reflectance_pattern)


# save initial parameter values in a dict (reference values)
ref_params = dict(ref_radiance_params)
ref_params.update(ref_reflectance_params)

# reference image
image_ref = mi.render(scene, spp=512)


# reassign selected scene parameters with random values
randomize_srgb_params(params, ref_radiance_params, 500.0)
randomize_srgb_params(params, ref_reflectance_params, 1.0)


# combine all reference scene parameter dict's
modified_params = dict(ref_radiance_params)
modified_params.update(ref_reflectance_params)

# print modified and reference scene parameter values
# for key in modified_params:
#     print('modified:', params[key])
#     print('ref:', modified_params[key])


# radiance optimizer
radiance_opt = mi.ad.Adam(
    lr=100.0, params={k: params[k] for k in ref_radiance_params})
params.update(radiance_opt)

# reflactence optimizer
reflectance_opt = mi.ad.Adam(
    lr=1.0, params={k: params[k] for k in ref_reflectance_params})
params.update(reflectance_opt)

# combine optimizators
opts = [radiance_opt, reflectance_opt]

# render initial image
img_init = mi.render(scene, spp=256)

# set initial parameter errors
param_errors = {k: [dr.sum(dr.sqr(ref_params[k] - params[k]))[0]]
                for k in modified_params}

# set optimization parameters
iteration_count = 200
min_errors = {k: 0.3 for k in ref_radiance_params}
min_errors.update({k: 0.01 for k in ref_reflectance_params})

for it in range(iteration_count):

    # check all optimization parameters and if defined threshold is
    # achieved stop optimization for that parameter (i.e. pop optimization param)
    for opt in opts:
        for key in list(opt.keys()):
            if key in opt and param_errors[key][-1] < min_errors[key]:
                opt.variables.pop(key)
                logging.info(f'Key {key} is optimized')

    # stop optimization if all optimization variables are empty
    # (i.e. if all optimization params reached a defined threshold)
    if all(map(lambda opt: not opt.variables, opts)):
        break

    # Perform a (noisy) differentiable rendering of the scene
    image = mi.render(scene, params, seed=it, spp=4)

    # Evaluate the objective function from the current rendered image
    loss = dr.sum(dr.sqr(image - image_ref)) / len(image)

    # Backpropagate through the rendering process
    dr.backward(loss)

    for opt in opts:
        # Optimizer: take a gradient descent step
        opt.step()
        for key in opt.keys():
            # Post-process the optimized parameters to ensure legal
            # radiance values
            if reflectance_pattern.search(key):
                opt[key] = dr.clamp(reflectance_opt[key], 0.0, 1.0)
        # Update the scene state to the new optimized values
        params.update(opt)

    # update errors that are being optimized
    print(
        f"Optimization status: {(it/iteration_count * 100):.1f}%", end='\r')
    logging.info(f"Iteration {it:02d}")
    for key in modified_params.keys():
        err = dr.sum(dr.sqr(ref_params[key] - params[key]))[0]
        param_errors[key].append(err)
        logging.info(f"\tkey= {key} error= {param_errors[key][-1]:6f}")
print('\nOptimization complete.')

# plot optimization results
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for k, v in param_errors.items():
    axs[0][0].plot(v, label=k)

axs[0][0].set_xlabel('iteration')
axs[0][0].set_ylabel('Loss')
axs[0][0].legend()
axs[0][0].set_title('Parameter error plot')

axs[0][1].imshow(mi.util.convert_to_bitmap(img_init))
axs[0][1].axis('off')
axs[0][1].set_title('Initial Image')

axs[1][0].imshow(mi.util.convert_to_bitmap(mi.render(scene, spp=512)))
axs[1][0].axis('off')
axs[1][0].set_title('Optimized image')

axs[1][1].imshow(mi.util.convert_to_bitmap(image_ref))
axs[1][1].axis('off')
axs[1][1].set_title('Reference Image')

plt.show()
fig.savefig('python\material-optimizer\material-optimizer-figure.png')
