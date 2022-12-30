import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi

from mitsuba.scalar_rgb import Transform4f as T

mi.set_variant("cuda_ad_rgb")


def showResults(sensors, ref_images, scene, init_images, loss_hist):
    for sensor_idx, sensor in enumerate(sensors):
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0][0].plot(loss_hist)
        axs[0][0].set_xlabel("Iteration")
        axs[0][0].set_ylabel("Loss")
        axs[0][0].set_title("Parameter error plot")

        axs[0][1].imshow(mi.util.convert_to_bitmap(init_images[sensor_idx]))
        axs[0][1].axis("off")
        axs[0][1].set_title("Initial Image")

        axs[1][0].imshow(
            mi.util.convert_to_bitmap(mi.render(scene, spp=512, sensor=sensor))
        )
        axs[1][0].axis("off")
        axs[1][0].set_title("Optimized image")

        axs[1][1].imshow(mi.util.convert_to_bitmap(ref_images[sensor_idx]))
        axs[1][1].axis("off")
        axs[1][1].set_title("Reference Image")

    plt.show()


def prepare_sensors(
    sensor_count,
    sensors,
    rotation_axis: list = [0, 1, 0],
    resolution: tuple = (256, 256),
):
    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i
        sensor_rotation = mi.ScalarTransform4f.rotate(rotation_axis, angle)
        sensor_to_world = mi.ScalarTransform4f.look_at(
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
                        "width": resolution[0],
                        "height": resolution[1],
                        "sample_border": True,
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
    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            # Perform the differentiable light transport simulation
            img = mi.render(
                scene,
                params,
                sensor=sensors[sensor_idx],
                spp=spp,
                seed=it + it * stage_count,
            )

            loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))

            # # Backpropagate gradients
            dr.backward(loss)

            # Take a gradient step
            opt.step()

            # clamp values
            opt[key] = dr.clamp(opt[key], -2.0, 2.0)

            # Propagate changes to the scene
            params.update(opt)

            total_loss += loss[0]

        if len(opt_hist) > 1 and total_loss > (loss_hist[-1] * 1.05):
            opt.reset(key)
            params.update(opt_hist[-1])
            total_loss = loss_hist[-1]
            learning_rate = max(0.00005, learning_rate - 0.00005)
            opt.set_learning_rate({key: learning_rate})
        else:
            opt_hist.append(opt)
            loss_hist.append(total_loss)
        print(f"Iteration {it:02d}: error={total_loss:6f}", end="\r")


scene_dict = {
    "type": "scene",
    "integrator": {"type": "direct_reparam"},
    "object": {
        "type": "ply",
        "filename": "scripts\\common\\meshes\\bunny.ply",
        "to_world": T.translate([0.2, -1.1, 0]).scale(13),
        "bsdf": {"type": "diffuse"},
    },
    "emitter": {"type": "envmap", "filename": "scripts\\common\\envmap.exr"},
}

scene_ref = mi.load_dict(scene_dict)

# Number of samples per pixel for reference images
sensor_count = 12
sensors = []
prepare_sensors(int(sensor_count/2), sensors, [1, 0, 0])
prepare_sensors(int(sensor_count/2), sensors)
ref_spp = 256
ref_images = [
    mi.render(scene_ref, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]

# Modify the scene dictionary
scene_dict["object"] = {
    "type": "obj",
    "filename": "scripts\\common\\meshes\\bunny_test.obj",
    "to_world": T.translate([0.2, -1.1, 0]).rotate([1, 0, 0], 90).scale(13),
    "bsdf": {"type": "diffuse"},
}

scene = mi.load_dict(scene_dict)
init_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]

params = mi.traverse(scene)
key = "object.vertex_positions"
learning_rate = 0.003
opt = mi.ad.Adam(lr=learning_rate)
opt[key] = params[key]
params.update(opt)
iteration_count = 100
stage_count = 5
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
    print(f"\nOptimization stage {i} completed.")
    showResults(sensors, ref_images, scene, init_images, loss_hist)

print("Optimization completed. Preparing results...")

final_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]

fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
if sensor_count == 1:
    plt.imshow(mi.util.convert_to_bitmap(final_images[0]))
else:
    for i in range(sensor_count):
        axs[i].imshow(mi.util.convert_to_bitmap(final_images[i]))
        axs[i].axis("off")
plt.show()

showResults(sensors, ref_images, scene, init_images, loss_hist)

print("Writing out resulted mesh...")
# Create an empty mesh (allocates buffers of the correct size)
resulted_mesh = mi.Mesh(
    "resulted_mesh",
    vertex_count=params["object.vertex_count"],
    face_count=params["object.face_count"],
    has_vertex_normals=False,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(resulted_mesh)
mesh_params["vertex_positions"] = dr.ravel(params[key])
mesh_params["faces"] = dr.ravel(params["object.faces"])
print(mesh_params.update())
resulted_mesh.write_ply("output/resulted_mesh.ply")
