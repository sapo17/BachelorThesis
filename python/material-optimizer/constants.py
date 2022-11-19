from pathlib import Path
import re


""" Configuration Constants """
IMAGES_DIR_PATH: str = "images/"
SCENES_DIR_PATH: str = "scenes/"
OUTPUT_DIR_PATH: str = "output/"
MY_APP_ID: str = "sapo.material-optimizer.0.1"  # arbitrary string
LOG_FILE: Path = Path("material-optimizer.log")
DEFAULT_MIN_ERR_ON_CUSTOM_IMG: float = 0.001
DEFAULT_ITERATION_COUNT_ON_CUSTOM_IMG: int = 50
SUPPORTED_SPP_VALUES: list = ["4", "16", "32", "64"]
CUDA_AD_RGB: str = "cuda_ad_rgb"
DEFAULT_MITSUBA_SCENE: str = "cbox.xml"
MITSUBA_PRB_INTEGRATOR: str = "prb"
MITSUBA_PRB_REPARAM_INTEGRATOR: str = "prb_reparam"
MITSUBA_PRBVOLPATH_INTEGRATOR: str = "prbvolpath"
COLUMN_LABEL_VALUE: str = "Value"
COLUMN_LABEL_LEARNING_RATE: str = "Learning Rate"
COLUMN_LABEL_MINIMUM_ERROR: str = "Minimum Error"
COLUMN_LABEL_ITERATION_COUNT: str = "Iteration Count"
COLUMN_LABEL_OPTIMIZE: str = "Optimize"
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
FIGURE_FILE_NAME: str = "material-optimizer-result-figure.png"
ITERATION_STRING: str = "iteration"
LOSS_STRING: str = "Loss"
PARAMETER_ERROR_PLOT_STRING: str = "Parameter error plot"
OFF_STRING: str = "off"
INITIAL_IMAGE_STRING: str = "Initial Image"
REFERENCE_IMAGE_STRING: str = "Reference Image"
XML_FILE_FILTER_STRING: str = "Xml File (*.xml)"
IMAGES_FILE_FILTER_STRING: str = "Images (*.png *.jpg)"
RESTART_OPTIMIZATION_STRING: str = "Restart Optimization"
OUTPUT_TO_JSON_STRING: str = "Output to JSON"


""" 
BSDF Parameter Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#
"""
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
# DATA_PATTERN: re.Pattern = re.compile(r".*\.data") currently not supported, requires support for tensor type
# TEXTURE_PATTERN: re.Pattern = re.compile(r".*\.texture") currently not supported, requires support for texture type
IRRADIANCE_PATTERN: re.Pattern = re.compile(
    r".*\.irradiance"
)  # supports only float/Color3f entry


""" 
Participating Media Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_media.html#
"""
ALBEDO_PATTERN: re.Pattern = re.compile(r".*\.albedo")
SIGMA_T_PATTERN: re.Pattern = re.compile(r".*\.sigma_t")

MAX_SIGMA_T_VALUE: float = 10.0


""" 
Phase functions Patterns Constants
See also: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_phase.html#phase-functions
"""
PHASE_G_PATTERN: re.Pattern = re.compile(r".*\.g")
MAX_PHASE_G_VALUE: float = 1.0
MIN_PHASE_G_VALUE: float = -1.0


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
]
PATTERNS_REQUIRE_VOLUMETRIC_INTEGRATOR = [ALBEDO_PATTERN, SIGMA_T_PATTERN]
