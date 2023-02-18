from pathlib import Path
import re
import numpy as np
import mitsuba as mi


""" Configuration Constants """
IMAGES_DIR_PATH: str = "images/"
SCENES_DIR_PATH: str = "scenes/"
SCENES_MESHES_DIR_PATH: str = SCENES_DIR_PATH + "meshes/"
OUTPUT_DIR_PATH: str = "output/"
MY_APP_ID: str = "sapo.material-optimizer.0.1"  # arbitrary string
LOG_FILE: Path = Path("material-optimizer.log")
DEFAULT_MIN_ERR: float = 0.001
DEFAULT_ITERATION_COUNT: int = 100
SUPPORTED_SPP_VALUES: list = np.array(2 ** np.arange(7)).astype(str)
DEFAULT_MITSUBA_SCENE: str = "cbox.xml"
MITSUBA_PRB_INTEGRATOR: str = "prb"
MITSUBA_PRB_REPARAM_INTEGRATOR: str = "prb_reparam"
MITSUBA_PRBVOLPATH_INTEGRATOR: str = "prbvolpath"
COLUMN_LABEL_VALUE: str = "Value"
COLUMN_LABEL_LEARNING_RATE: str = "Learning Rate"
COLUMN_LABEL_MINIMUM_ERROR: str = "Minimum Error"
COLUMN_LABEL_ITERATION_COUNT: str = "Iteration Count"
COLUMN_LABEL_OPTIMIZE: str = "Optimize"
COLUMN_LABEL_MIN_CLAMP_LABEL: str = "Min. Clamp Value"
COLUMN_LABEL_MAX_CLAMP_LABEL: str = "Max. Clamp Value"
WINDOW_ICON_FILE_NAME: str = "sloth.png"
MATERIAL_OPTIMIZER_STRING: str = "Material Optimizer"
IMPORT_STRING: str = "Import"
IMPORT_SHORTCUT_STRING: str = "Ctrl+I"
IMPORT_LABEL_STRING: str = "Import Mitsuba 3 Scene File"
FILE_STRING: str = "&File"
IMPORT_FILE_STRING: str = "Import File"
INFO_STRING: str = "Info"
START_OPTIMIZATION_STRING: str = "Start Optimization"
NOT_IMPLEMENTED_STRING: str = "Not implemented yet"
LAST_ITERATION_STRING: str = "Last Iteration"
ITERATION_STRING: str = "iteration"
LOSS_STRING: str = "Loss"
PARAMETER_ERROR_PLOT_STRING: str = "Parameter error plot"
OFF_STRING: str = "off"
INITIAL_IMAGE_STRING: str = "Initial Image"
REFERENCE_IMAGE_STRING: str = "Reference Image"
XML_FILE_FILTER_STRING: str = "Xml File (*.xml)"
IMAGES_FILE_FILTER_STRING: str = "Images (*.png *.jpg)"
RESTART_OPTIMIZATION_STRING: str = "Restart Optimization"
OUTPUT_TO_JSON_STRING: str = "Output results"
MSE_STRING: str = "Mean Squared Error (MSE, L2 Loss)"
BRIGHTNESS_IDP_MSE_STRING: str = "Brightness Independent Mean Squared Error"
DUAL_BUFFER_STRING: str = "Deng et al. Dual Buffer Method"
MAE_STRING: str = "Mean Absolute Error (MAE, L1 Loss)"
MBE_STRING: str = "Mean Bias Error (MBE)"
LOSS_FUNCTION_STRINGS: list = [
    MSE_STRING,
    BRIGHTNESS_IDP_MSE_STRING,
    DUAL_BUFFER_STRING,
    MAE_STRING,
    MBE_STRING,
]
LOSS_FUNCTION_STRING: str = "Loss function"
SPP_DURING_OPT_STRING: str = "Samples per pixel"
DEFAULT_MIN_CLAMP_VALUE: float = 0.001
DEFAULT_MAX_CLAMP_VALUE: float = 0.999
LOAD_REF_IMG_LABEL: str = "Load reference image/s"
DEFAULT_LEARNING_RATE: float = 0.03
MAX_LEARNING_RATE: float = 0.9
MIN_LEARNING_RATE: float = 0.0001
SENSOR_IDX_LABEL: str = "Sensor Index"
INF_STR = "inf"
CLOSE_STATUS_STR = "CLOSE"
INITIAL_STATUS_STR = "INITIAL"
RENDER_STATUS_STR = "RENDER"
MARGIN_PERCENTAGE_LABEL = "Margin per update"
MARGIN_PENALTY_LABEL = "Margin Penalty"
NONE_STR = "None"
EXPONENTIAL_DECAY_STR = "Exponential Decay"

""" 
BSDF Parameter Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#
"""
EMPTY_PATTERN: re.Pattern = re.compile(r"")
REFLECTANCE_PATTERN: re.Pattern = re.compile(r".*\.reflectance")
ETA_PATTERN: re.Pattern = re.compile(r".*\.eta")
ALPHA_PATTERN: re.Pattern = re.compile(r".*\.alpha")
BASE_COLOR_PATTERN: re.Pattern = re.compile(r".*\.base_color")
ROUGHNESS_PATTERN: re.Pattern = re.compile(r".*\.roughness")
DIFF_TRANS_PATTERN: re.Pattern = re.compile(r".*\.diff_trans")
SPECULAR_REFLECTANCE_PATTERN: re.Pattern = re.compile(
    r".*\.specular_reflectance"
)
SPECULAR_TRANSMITTANCE_PATTERN: re.Pattern = re.compile(
    r".*\.specular_transmittance"
)
K_PATTERN: re.Pattern = re.compile(r".*\.k")
DIFFUSE_REFLECTANCE_PATTERN: re.Pattern = re.compile(
    r".*\.diffuse_reflectance"
)
WEIGHT_PATTERN: re.Pattern = re.compile(r".*\.weight")
OPACITY_PATTERN: re.Pattern = re.compile(r".*\.opacity")
THETA_PATTERN: re.Pattern = re.compile(r".*\.theta")
TRANSMITTANCE_PATTERN: re.Pattern = re.compile(r".*\.transmittance")
DELTA_PATTERN: re.Pattern = re.compile(r".*\.delta")
ANISOTROPIC_PATTERN: re.Pattern = re.compile(r".*\.anisotropic")
METALLIC_PATTERN: re.Pattern = re.compile(r".*\.metallic")
SPEC_TRANS_PATTERN: re.Pattern = re.compile(r".*\.spec_trans")
SPECULAR_PATTERN: re.Pattern = re.compile(r".*\.specular")
SPEC_TINT_PATTERN: re.Pattern = re.compile(r".*\.spec_tint")
SHEEN_PATTERN: re.Pattern = re.compile(r".*\.sheen")
SHEEN_TINT_PATTERN: re.Pattern = re.compile(r".*\.sheen_tint")
FLATNESS_PATTERN: re.Pattern = re.compile(r".*\.flatness")
CLEARCOAT_PATTERN: re.Pattern = re.compile(r".*\.clearcoat")
CLEARCOAT_GLOSS_PATTERN: re.Pattern = re.compile(r".*\.clearcoat_gloss")

# max Index of refraction value is taken from
# https://en.wikipedia.org/wiki/List_of_refractive_indices
# Germanium: ~4.1
MAX_ETA_VALUE: float = 4.1
MAX_K_VALUE: float = 4.1


"""
To get an intuition about the effect of the surface roughness parameter , 
consider the following approximate classification: a value of [0.001, 0.01] 
corresponds to a material with slight imperfections on an otherwise smooth 
surface finish, 0.1 is relatively rough, and [0.3, 0.7] is extremely rough 
(e.g. an etched or ground finish). Values significantly above that are probably 
not too realistic.
https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#rough-dielectric-material-roughdielectric
"""
MAX_ALPHA_VALUE: float = 0.7

# taken from https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#the-thin-principled-bsdf-principledthin
MAX_DIFF_TRANS_VALUE: float = 2.0

# taken from https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#linear-retarder-material-retarder
MAX_DELTA_VALUE: float = 360.0


""" 
Emitter Parameter Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_emitters.html
"""
RADIANCE_PATTERN: re.Pattern = re.compile(r".*\.radiance")
INTENSITY_PATTERN: re.Pattern = re.compile(r".*\.intensity")
SCALE_PATTERN: re.Pattern = re.compile(r".*\.scale")
IRRADIANCE_PATTERN: re.Pattern = re.compile(
    r".*\.irradiance"
)  # supports only float/Color3f entry

# TODO: what can be the max radiance value
MAX_RADIANCE_VALUE: float = 10000.0 
MAX_SCALE_VALUE: int = 100


""" 
Participating Media Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#
"""
ALBEDO_PATTERN: re.Pattern = re.compile(r".*\.albedo")
SIGMA_T_PATTERN: re.Pattern = re.compile(r".*\.sigma_t")


""" 
Phase functions Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#phase-functions
"""
PHASE_G_PATTERN: re.Pattern = re.compile(r".*\.g")
MAX_PHASE_G_VALUE: float = 1.0
MIN_PHASE_G_VALUE: float = -1.0


""" 
Texture Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html#
"""
VERTEX_COLOR_PATTERN: re.Pattern = re.compile(r".*\.vertex_(Col|color)")

""" 
Grid-based volume data source Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_volumes.html#volumes
"""
ALBEDO_DATA_PATTERN: re.Pattern = re.compile(r".*\.albedo.data")

""" 
Shapes Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html#
"""
VERTEX_POSITIONS_PATTERN: re.Pattern = re.compile(r".*\.vertex_positions")
# TODO: what can be the min/max vertex position value?
MAX_VERTEX_POSITION_VALUE: mi.Point3f = mi.Point3f(100.0, 100.0, 100.0)

""" Combine Patterns """
SUPPORTED_MITSUBA_PARAMETER_PATTERNS: list = [
    REFLECTANCE_PATTERN,
    RADIANCE_PATTERN,
    ETA_PATTERN,
    ALPHA_PATTERN,
    BASE_COLOR_PATTERN,
    ROUGHNESS_PATTERN,
    DIFF_TRANS_PATTERN,
    SPECULAR_REFLECTANCE_PATTERN,
    SPECULAR_TRANSMITTANCE_PATTERN,
    K_PATTERN,
    DIFFUSE_REFLECTANCE_PATTERN,
    WEIGHT_PATTERN,
    OPACITY_PATTERN,
    THETA_PATTERN,
    TRANSMITTANCE_PATTERN,
    DELTA_PATTERN,
    ANISOTROPIC_PATTERN,
    METALLIC_PATTERN,
    SPEC_TRANS_PATTERN,
    SPECULAR_PATTERN,
    SPEC_TINT_PATTERN,
    SHEEN_PATTERN,
    SHEEN_TINT_PATTERN,
    FLATNESS_PATTERN,
    CLEARCOAT_PATTERN,
    CLEARCOAT_GLOSS_PATTERN,
    INTENSITY_PATTERN,
    SCALE_PATTERN,
    IRRADIANCE_PATTERN,
    ALBEDO_PATTERN,
    SIGMA_T_PATTERN,
    PHASE_G_PATTERN,
    VERTEX_COLOR_PATTERN,
    ALBEDO_DATA_PATTERN,
    VERTEX_POSITIONS_PATTERN,
]
PATTERNS_INTRODUCE_DISCONTINUITIES: list = [
    # see also parameter 'D' flags https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#technical-details
    ETA_PATTERN,
    ALPHA_PATTERN,
    ROUGHNESS_PATTERN,
    OPACITY_PATTERN,
    THETA_PATTERN,
    ANISOTROPIC_PATTERN,
    METALLIC_PATTERN,
    SPEC_TRANS_PATTERN,
    SPECULAR_PATTERN,
    CLEARCOAT_PATTERN,
    CLEARCOAT_GLOSS_PATTERN,
    PHASE_G_PATTERN,
    VERTEX_POSITIONS_PATTERN,
]
PATTERNS_REQUIRE_VOLUMETRIC_INTEGRATOR = [ALBEDO_PATTERN, SIGMA_T_PATTERN]

### Constant dictionary: key: Pattern, value: default min and max clamp value
def getDefaultLegalValues(pattern: re.Pattern) -> tuple([int, int]):
    result = (DEFAULT_MIN_CLAMP_VALUE, DEFAULT_MAX_CLAMP_VALUE)

    if pattern is ETA_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_ETA_VALUE)
    elif pattern is K_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_K_VALUE)
    elif pattern is ALPHA_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_ALPHA_VALUE)
    elif pattern is DIFF_TRANS_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_DIFF_TRANS_VALUE)
    elif pattern is DELTA_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_DELTA_VALUE)
    elif pattern is PHASE_G_PATTERN:
        result = (MIN_PHASE_G_VALUE, MAX_PHASE_G_VALUE)
    elif pattern is SCALE_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_SCALE_VALUE)
    elif pattern is RADIANCE_PATTERN:
        result = (DEFAULT_MIN_CLAMP_VALUE, MAX_RADIANCE_VALUE)
    elif pattern is VERTEX_POSITIONS_PATTERN:
        # currently arbitrary: more or less user responsibility
        result = (-MAX_VERTEX_POSITION_VALUE, MAX_VERTEX_POSITION_VALUE)

    return result


DEFAULT_CLAMP_VALUES: dict = {
    pattern: getDefaultLegalValues(pattern)
    for pattern in SUPPORTED_MITSUBA_PARAMETER_PATTERNS
}

""" Scene Constants """
CBOX_SCENE_PATH = SCENES_DIR_PATH + DEFAULT_MITSUBA_SCENE
CBOX_LUMINAIRE_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_luminaire.obj"
CBOX_FLOOR_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_floor.obj"
CBOX_CEILING_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_ceiling.obj"
CBOX_BACK_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_back.obj"
CBOX_GREENWALL_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_greenwall.obj"
CBOX_REDWALL_OBJ_PATH: str = SCENES_MESHES_DIR_PATH + "cbox_redwall.obj"

CBOX_LUMINAIRE_OBJ_STRING = """
v  0.25  1 -0.25
v  0.25  1  0.25
v -0.25  1  0.25
v -0.25  1 -0.25
f 1 2 3 4
"""

CBOX_BACK_OBJ_STRING = """
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1 -1
f 1 2 3 4
"""

CBOX_CEILING_OBJ_STRING = """
v  1  1 -1
v  1  1  1
v -1  1  1
v -1  1 -1
f 1 2 3 4
"""

CBOX_FLOOR_OBJ_STRING = """
v -1 -1  1
v  1 -1  1
v  1 -1 -1
v -1 -1 -1
f 1 2 3 4
"""

CBOX_GREENWALL_OBJ_STRING = """
v -1  1 -1
v -1  1  1
v -1 -1  1
v -1 -1 -1
f 1 2 3 4
"""

CBOX_REDWALL_OBJ_STRING = """
v 1 -1  1
v 1  1  1
v 1  1 -1
v 1 -1 -1
f 1 2 3 4
"""

CBOX_XML_STRING = """
<scene version="3.0.0">
    <default name="spp" value="128"/>
    <default name="resx" value="256"/>
    <default name="resy" value="256"/>
    <default name="max_depth" value="6"/>
    <default name="integrator" value="path"/>
    <default name="sample_border" value="true"/>

    <integrator type='$integrator'>
        <integer name="max_depth" value="$max_depth"/>
    </integrator>

    <sensor type="perspective" id="sensor">
        <string name="fov_axis" value="smaller"/>
        <float name="near_clip" value="0.001"/>
        <float name="far_clip" value="100.0"/>
        <float name="focus_distance" value="1000"/>
        <float name="fov" value="39.3077"/>
        <transform name="to_world">
            <lookat origin="0,  0,  4"
                    target="0,  0,  0"
                    up    ="0,  1,  0"/>
        </transform>
        <sampler type="independent">
            <integer name="sample_count" value="$spp"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width"  value="$resx"/>
            <integer name="height" value="$resy"/>
            <rfilter type="tent"/>
            <string name="pixel_format" value="rgb"/>
            <string name="component_format" value="float32"/>
            <boolean name="sample_border" value="$sample_border"/>
        </film>
    </sensor>

    <!-- BSDFs -->

    <bsdf type="diffuse" id="gray">
        <rgb name="reflectance" value="0.85, 0.85, 0.85"/>
    </bsdf>

    <bsdf type="diffuse" id="white">
        <rgb name="reflectance" value="0.885809, 0.698859, 0.666422"/>
    </bsdf>

    <bsdf type="diffuse" id="green">
        <rgb name="reflectance" value="0.105421, 0.37798, 0.076425"/>
    </bsdf>

    <bsdf type="diffuse" id="red">
        <rgb name="reflectance" value="0.570068, 0.0430135, 0.0443706"/>
    </bsdf>

    <bsdf type="dielectric" id="glass"/>

    <bsdf type="conductor" id="mirror"/>

    <!-- Light -->

    <shape type="obj" id="light">
        <string name="filename" value="meshes/cbox_luminaire.obj"/>
        <transform name="to_world">
            <translate x="0" y="-0.01" z="0"/>
        </transform>
        <ref id="white"/>
        <emitter type="area">
            <rgb name="radiance" value="18.387, 13.9873, 6.75357"/>
        </emitter>
    </shape>

    <!-- Shapes -->

    <shape type="obj" id="floor">
        <string name="filename" value="meshes/cbox_floor.obj"/>
        <ref id="white"/>
    </shape>

    <shape type="obj" id="ceiling">
        <string name="filename" value="meshes/cbox_ceiling.obj"/>
        <ref id="white"/>
    </shape>

    <shape type="obj" id="back">
        <string name="filename" value="meshes/cbox_back.obj"/>
        <ref id="white"/>
    </shape>

    <shape type="obj" id="greenwall">
        <string name="filename" value="meshes/cbox_greenwall.obj"/>
        <ref id="green"/>
    </shape>

    <shape type="obj" id="redwall">
        <string name="filename" value="meshes/cbox_redwall.obj"/>
        <ref id="red"/>
    </shape>

    <shape type="sphere" id="mirrorsphere">
        <transform name="to_world">
            <scale value="0.5"/>
            <translate x="-0.3" y="-0.5" z="0.2"/>
        </transform>
        <ref id="mirror"/>
    </shape>

    <shape type="sphere" id="glasssphere">
        <transform name="to_world">
            <scale value="0.25"/>
            <translate x="0.5" y="-0.75" z="-0.2"/>
        </transform>
        <ref id="glass"/>
    </shape>
</scene>
"""
