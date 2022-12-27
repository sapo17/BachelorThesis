import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T

mi.set_variant('cuda_ad_rgb')

sensor_count = 8
sensors = []

def prepare_sensors(sensor_count, sensors, resolution: tuple):
    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensor_rotation = mi.ScalarTransform4f.rotate([0, 1, 0], angle)
        sensor_to_world = mi.ScalarTransform4f.look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
        sensors.append(mi.load_dict({
        'type': 'perspective',
        'fov': 45,
        'to_world': sensor_rotation @ sensor_to_world,
        'film': {
            'type': 'hdrfilm',
            'width': resolution[0], 'height': resolution[1],
            'sample_border': True
        },
        'sampler': {
            "type": "multijitter",
        }
    }))


def optimize(sensor_count, sensors, ref_images, scene, params, key, opt, iteration_count, spp, loss_hist):
    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            # Perform the differentiable light transport simulation
            img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it)

            # L2 loss function
            loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))

            # Backpropagate gradients
            dr.backward(loss)

            # Take a gradient step
            opt.step()

            # Clamp the optimized density values. Since we used the `scale` parameter
            # when instantiating the volume, we are in fact optimizing extinction
            # in a range from [1e-6 * scale, scale].
            opt[key] = dr.clamp(opt[key], -10.0, 10.0)

            # Propagate changes to the scene
            params.update(opt)

            total_loss += loss[0]
        print(f"Iteration {it:02d}: error={total_loss:6f}", end='\r')
        if total_loss < 0.01:
            break
        loss_hist.append(total_loss)


scene_dict = {
    'type': 'scene',
    'integrator': {'type': 'direct_reparam'},
    'object': {
        'type': 'ply',
        'filename': 'scripts\\common\\meshes\\bunny.ply',
        'to_world': T.translate([0, -1, 0]).scale(12),
        'bsdf': {'type': 'diffuse'},
    },
    'emitter': {
        'type': 'envmap',
        'filename': 'scripts\\common\\envmap.exr'
    }
}

scene_ref = mi.load_dict(scene_dict)

# Number of samples per pixel for reference images
prepare_sensors(sensor_count, sensors, (512, 512))
ref_spp = 256
ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

# Modify the scene dictionary
scene_dict['object'] = {
    'type': 'ply',
    'filename': 'scripts\\common\\meshes\\Sphere.ply',
    'to_world': T.scale(0.5),
    'bsdf': {'type': 'diffuse'}
}

scene = mi.load_dict(scene_dict)
init_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

params = mi.traverse(scene)
key = 'object.vertex_positions'
learning_rate = 0.003
opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt);
iteration_count = 10
stage_count = 5
spp = 4
loss_hist = []

prepare_sensors(sensor_count, sensors, (256, 256))
for i in range(stage_count):
    optimize(sensor_count, sensors, ref_images, scene, params, key, opt, iteration_count, spp, loss_hist)
    print(f"\nOptimization stage {i} completed.")

print("Optimization completed. Preparing results...")

final_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]
fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
for i in range(sensor_count):
    axs[i].imshow(mi.util.convert_to_bitmap(final_images[i]))
    axs[i].axis('off')
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(loss_hist)
axs[0][0].set_xlabel('Iteration');
axs[0][0].set_ylabel('Loss');
axs[0][0].set_title('Parameter error plot');

axs[0][1].imshow(mi.util.convert_to_bitmap(init_images[0]))
axs[0][1].axis('off')
axs[0][1].set_title('Initial Image')

axs[1][0].imshow(mi.util.convert_to_bitmap(mi.render(scene, spp=512, sensor=sensors[0])))
axs[1][0].axis('off')
axs[1][0].set_title('Optimized image')

axs[1][1].imshow(mi.util.convert_to_bitmap(ref_images[0]))
axs[1][1].axis('off')
axs[1][1].set_title('Reference Image');
plt.show()

