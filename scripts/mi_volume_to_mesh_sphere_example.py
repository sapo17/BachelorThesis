import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant("llvm_ad_rgb")

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


grid_res = 50
x, y, z = np.ogrid[
    0 : 1 : grid_res * 1j, 0 : 1 : grid_res * 1j, 0 : 1 : grid_res * 1j
]
sphere = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.4**2
sphere = sphere.astype(np.float32)
sphere = sphere[..., np.newaxis]

scene_dict = {
    "type": "scene",
    "integrator": {"type": "prbvolpath", "max_depth": 64},
    "object": {
        "type": "cube",
        "interior": {
            "type": "heterogeneous",
            "sigma_t": {
                "type": "gridvolume",
                "grid": mi.VolumeGrid(mi.TensorXf(sphere)),
                "to_world": T.translate(-1).scale(2.0),
            },
            "scale": 100.0,
        },
        "bsdf": {"type": "null"},
    },
    "emitter": {"type": "envmap", "filename": "scripts/common/envmap.exr"},
}

scene = mi.load_dict(scene_dict)
sensor_count = 1
sensors = []
prepare_sensors(sensor_count, sensors)
ref_spp = 128
ref_images = [
    mi.render(scene, sensor=sensors[i], spp=ref_spp)
    for i in range(sensor_count)
]
img = mi.util.convert_to_bitmap(ref_images[0])
plt.imshow(img)
plt.tight_layout()
plt.show()

params = mi.traverse(scene)

# applying marching cubes
verts, faces, normals, values = measure.marching_cubes(
    np.array(params["object.interior_medium.sigma_t.data"])[:, :, :, 0], 0
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor("k")
ax.add_collection3d(mesh)

ax.set_xlim(0, grid_res)
ax.set_ylim(0, grid_res)
ax.set_zlim(0, grid_res)

plt.tight_layout()
plt.show()
verts, faces = convert_obj_to_br(verts, faces, 50)
marching_cubes_to_obj(
    (verts, faces, normals, values), "output\\test_marching_cubes.obj"
)
