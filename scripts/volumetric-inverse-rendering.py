import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

from mitsuba import ScalarTransform4f as T


def prepare_sensors(sensor_count, sensors):
    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensor_rotation = T.rotate([0, 1, 0], angle)
        sensor_to_world = T.look_at(
            target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0]
        )
        sensors.append(
            mi.load_dict(
                {
                    "type": "perspective",
                    "fov": 45,
                    "to_world": sensor_rotation @ sensor_to_world,
                    "film": {
                        "type": "hdrfilm",
                        "width": 256,
                        "height": 256,
                        "filter": {"type": "gaussian"},
                    },
                }
            )
        )


def optimize(
    sensor_count,
    sensors,
    ref_images,
    scene,
    params,
    key,
    opt,
    iteration_count,
    spp,
    loss_hist,
    opt_hist,
    learning_rate,
):
    amount_of_unsuccessfull_steps = 0
    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            # Perform the differentiable light transport simulation
            img = mi.render(
                scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it
            )

            # L2 loss function
            loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))

            # Backpropagate gradients
            dr.backward(loss)

            # Take a gradient step
            opt.step()

            # Clamp the optimized density values. Since we used the `scale` parameter
            # when instantiating the volume, we are in fact optimizing extinction
            # in a range from [1e-6 * scale, scale].
            opt[key] = dr.clamp(opt[key], 1e-6, 1.0)

            # Propagate changes to the scene
            params.update(opt)

            total_loss += loss[0]
        if len(opt_hist) > 1 and total_loss > (
            loss_hist[-1] + loss_hist[-1] * 0.3
        ):
            amount_of_unsuccessfull_steps += 1
            opt.reset(key)
            params.update(opt_hist[-1])
            total_loss = loss_hist[-1]
            if amount_of_unsuccessfull_steps % 5 == 0:
                learning_rate = learning_rate / 1.1
                opt.set_learning_rate({key: learning_rate})
        else:
            opt_hist.append(opt)
            loss_hist.append(total_loss)
        print(f"Iteration {it:02d}: error={total_loss:6f}", end="\r")
        loss_hist.append(total_loss)


scene_dict = {
    "type": "scene",
    "integrator": {"type": "prbvolpath"},
    "object": {
        "type": "ply",
        "filename": "scripts/common/meshes/bunny.ply",
        "to_world": T.translate([0, -1.25, 0]).scale(13),
        "bsdf": {"type": "diffuse"},
    },
    "emitter": {"type": "envmap", "filename": "scripts/common/envmap.exr"},
}

scene_ref = mi.load_dict(scene_dict)

sensor_count = 7
sensors = []

prepare_sensors(sensor_count, sensors)
ref_spp = 256
ref_images = [
    mi.render(scene_ref, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]
grid_res = 16

# Modify the scene dictionary
scene_dict["object"] = {
    "type": "cube",
    "interior": {
        "type": "heterogeneous",
        "sigma_t": {
            "type": "gridvolume",
            "grid": mi.VolumeGrid(
                dr.full(mi.TensorXf, 0.002, (grid_res, grid_res, grid_res, 1))
            ),
            "to_world": T.translate(-1).scale(2.0),
        },
        "scale": 40.0,
    },
    "bsdf": {"type": "null"},
}

scene = mi.load_dict(scene_dict)
init_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]

params = mi.traverse(scene)
key = "object.interior_medium.sigma_t.data"
learning_rate = 0.003
opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt)
iteration_count = 15
stage_count = 4
spp = 4
loss_hist = []
opt_hist = []

for i in range(stage_count):
    optimize(
        sensor_count,
        sensors,
        ref_images,
        scene,
        params,
        key,
        opt,
        iteration_count,
        spp,
        loss_hist,
        opt_hist,
        learning_rate,
    )
    grid_res = grid_res * 2
    opt[key] = dr.upsample(opt[key], shape=(grid_res, grid_res, grid_res))
    params.update(opt)
    print(f"\nOptimization stage {i} completed.")

print("Optimization completed. Preparing results...")

final_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]

fig, axs = plt.subplots(2, sensor_count, figsize=(14, 6))
for i in range(sensor_count):
    axs[0][i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    axs[0][i].axis("off")
    axs[1][i].imshow(mi.util.convert_to_bitmap(final_images[i]))
    axs[1][i].axis("off")
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].plot(loss_hist)
axs[0][0].set_xlabel("Iteration")
axs[0][0].set_ylabel("Loss")
axs[0][0].set_title("Parameter error plot")

axs[0][1].imshow(mi.util.convert_to_bitmap(init_images[0]))
axs[0][1].axis("off")
axs[0][1].set_title("Initial Image")

axs[1][0].imshow(
    mi.util.convert_to_bitmap(mi.render(scene, spp=512, sensor=sensors[0]))
)
axs[1][0].axis("off")
axs[1][0].set_title("Optimized image")

axs[1][1].imshow(mi.util.convert_to_bitmap(ref_images[0]))
axs[1][1].axis("off")
axs[1][1].set_title("Reference Image")
plt.show()

# output = f"output/volumetric_inverse_rendering_result.vol"
# mi.VolumeGrid(params[key]).write(output)

# apply marching cubes
