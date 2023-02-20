import numpy as np

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

m = mi.load_string("""
        <shape type="ply" version="0.5.0">
            <string name="filename" value="scenes\metashape\iliad\meshes\Iliad-orig.ply"/>
        </shape>
    """)

m.add_attribute("vertex_color", 3, np.load('output\\vertex_color_numpy_array_PLYMesh.vertex_color_iteration_29_2022-12-05_11_44_18.npy'))
m.write_ply("test.ply")