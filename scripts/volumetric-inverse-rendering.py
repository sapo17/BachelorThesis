import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")

from mitsuba import ScalarTransform4f as T
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def convert_obj_to_br(verts, faces, voxel_size):
    """
    Note from Hasbay: Code taken from:
    https://programtalk.com/vs4/python/brainglobe/brainreg-segment/brainreg_segment/regions/IO.py/
    """
    if voxel_size != 1:
        verts = verts * voxel_size

    faces = faces + 1
    return verts, faces


def marching_cubes_to_obj(marching_cubes_out, output_file):
    """
    Note from Hasbay: Code taken from:
    https://programtalk.com/vs4/python/brainglobe/brainreg-segment/brainreg_segment/regions/IO.py/
    Saves the output of skimage.measure.marching_cubes as an .obj file
    :param marching_cubes_out: tuple
    :param output_file: str
    """

    verts, faces, normals, _ = marching_cubes_out
    with open(output_file, "w") as f:
        for item in verts:
            f.write(f"v {item[0]} {item[1]} {item[2]}\n")
        for item in normals:
            f.write(f"vn {item[0]} {item[1]} {item[2]}\n")
        for item in faces:
            f.write(
                f"f {item[0]}//{item[0]} {item[1]}//{item[1]} "
                f"{item[2]}//{item[2]}\n"
            )
        f.close()


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
                    "fov": 35,
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


def exponentialDecay(original: float, decayFactor: float, time: int):
    """time = iteration in our case"""
    return original * (1 - decayFactor) ** time


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
    margin,
):
    amount_of_unsuccessfull_steps = 0
    new_learning_rate = learning_rate
    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            # Perform the differentiable light transport simulation
            img = mi.render(
                scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it
            )

            # compute L1 loss
            loss = dr.mean(dr.abs(img - ref_images[sensor_idx]))
            # loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))

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
            loss_hist[-1] + loss_hist[-1] * margin
        ):
            amount_of_unsuccessfull_steps += 1
            opt.set_learning_rate({key: new_learning_rate})
            if amount_of_unsuccessfull_steps % 3 == 0:
                new_learning_rate = max(
                    0.00005, exponentialDecay(learning_rate, 0.03, it)
                )
                print(f"New learning rate: {new_learning_rate}")
                opt.reset(key)
        else:
            opt_hist.append(opt)
            loss_hist.append(total_loss)
        print(f"Iteration {it:02d}: error={total_loss:6f}", end="\r")
        loss_hist.append(total_loss)

    return new_learning_rate


scene_dict = {
    "type": "scene",
    "integrator": {"type": "prbvolpath", "max_depth": 2, "rr_depth": 1},
    "object": {
        "type": "ply",
        "filename": "scenes\material-preview\\translucent-principled-textured-dragon\meshes\Dragon.ply",
        # "filename": "scripts/common/meshes/bunny.ply",
        # "to_world": T.translate([0.2, -1.3, 0]).scale(11),
        "to_world": T.translate([0, -1, 0]).scale(0.4),
        "bsdf": {"type": "diffuse"},
    },
    "emitter": {"type": "constant"},
    # "emitter": {"type": "envmap", "filename": "scripts/common/envmap.exr"},
}

scene_ref = mi.load_dict(scene_dict)

sensor_count = 8
sensors = []

prepare_sensors(sensor_count, sensors)
ref_spp = 8
ref_images = [
    mi.render(scene_ref, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]
grid_res = 8

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
            "to_world": T.translate(-1).scale(2),
        },
        "scale": 100.0,
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
learning_rate = 0.001
opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt)
iteration_count = 8
stage_count = 5
spp = 1
loss_hist = []
opt_hist = []
margin = 0.0
max_res = 64
max_iteration_per_step = 64
contour_value_for_isosurfaces_in_volume = 0.005

for i in range(stage_count):
    print(f"Begin optimization stage {i}.")
    learning_rate = optimize(
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
        margin,
    )
    grid_res = min(max_res, grid_res * 2)
    iteration_count = min(max_iteration_per_step, iteration_count * 2)
    print(f"New configuration: ic={iteration_count}, res={grid_res}")
    opt[key] = dr.upsample(opt[key], shape=(grid_res, grid_res, grid_res))
    params.update(opt)
    print(f"\nOptimization stage {i} completed.")

print("Optimization completed. Preparing results...")

final_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(2)
]

fig, axs = plt.subplots(2, 2)
for i in range(2):
    axs[0][i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    axs[0][i].axis("off")
    axs[1][i].imshow(mi.util.convert_to_bitmap(final_images[i]))
    axs[1][i].axis("off")
fig.tight_layout()
fig.savefig(
    "output/volumetric_inverse_rendering_images.png",
    pad_inches=0,
    bbox_inches="tight",
    transparent=True,
)
plt.show()

plt.plot(loss_hist)
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.title(f"Image loss, final: {loss_hist[-1]:3f}")
plt.tight_layout()
plt.savefig(
    "output/volumetric_inverse_rendering_loss.png",
    pad_inches=0,
    bbox_inches="tight",
)
plt.show()

output = f"output/volumetric_inverse_rendering_result.vol"
mi.VolumeGrid(opt[key]).write(output)

# apply marching cubes
verts, faces, normals, values = measure.marching_cubes(
    np.array(opt[key])[:, :, :, 0],
    contour_value_for_isosurfaces_in_volume,
    allow_degenerate=False,
)

verts, faces = convert_obj_to_br(verts, faces, grid_res)
marching_cubes_to_obj(
    (verts, faces, normals, values), "output\\volumetric_inverse_rendering.obj"
)
