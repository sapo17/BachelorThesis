import numpy as np

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

m = mi.load_string("""
        <shape type="ply" version="0.5.0">
            <string name="filename" value="scenes\metashape\mini-birdy-statue\multiple_sensors\meshes\icosphere.ply"/>
        </shape>
    """)

m.add_attribute("vertex_color", 3,  np.zeros(3 * m.vertex_count()))
m.write_ply("icosphere_vc_orig.ply")