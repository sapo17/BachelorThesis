import numpy as np
import matplotlib.pyplot as plt

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

x, y, z = np.ogrid[0:1:50j, 0:1:50j, 0:1:50j]
sphere = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.4**2
sphere = sphere.astype(np.float32)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(sphere, 0)

# # Display resulting triangular mesh using Matplotlib. This can also be done
# # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

# ax.set_xlim(0, 50)
# ax.set_ylim(0, 50)
# ax.set_zlim(0, 50)

# plt.tight_layout()
# plt.show()
verts, faces = convert_obj_to_br(verts, faces, 50)
marching_cubes_to_obj(
    (verts, faces, normals, values), "output\\test_marching_cubes.obj"
)
