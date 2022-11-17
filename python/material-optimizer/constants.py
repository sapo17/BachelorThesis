from pathlib import Path
import re

IMAGES_DIR_PATH = "images/"
SCENES_DIR_PATH = "scenes/"
MY_APP_ID = "sapo.material-optimizer.0.1"  # arbitrary string
REFLECTANCE_PATTERN: re.Pattern = re.compile(r".*\.reflectance\.value")
RADIANCE_PATTERN: re.Pattern = re.compile(r".*\.radiance\.value")
ETA_PATTERN: re.Pattern = re.compile(r".*\.eta")
ALPHA_PATTERN: re.Pattern = re.compile(r".*\.alpha")
BASE_COLOR_PATTERN: re.Pattern = re.compile(r".*\.base_color")
ROUGHNESS_PATTERN: re.Pattern = re.compile(r".*\.roughness")
DIFF_TRANS_PATTERN: re.Pattern = re.compile(r".*\.diff_trans")
SUPPORTED_BSDF_PATTERNS = [
    REFLECTANCE_PATTERN,
    RADIANCE_PATTERN,
    ETA_PATTERN,
    ALPHA_PATTERN,
    BASE_COLOR_PATTERN,
    ROUGHNESS_PATTERN,
    DIFF_TRANS_PATTERN,
]
PATTERNS_INTRODUCE_DISCONTINUITIES = [
    # see also parameter 'D' flags https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#technical-details
    ETA_PATTERN,
    ALPHA_PATTERN,
    ROUGHNESS_PATTERN,
]
LOG_FILE = Path("material-optimizer.log")
DEFAULT_MIN_ERR_ON_CUSTOM_IMG = 0.001
SUPPORTED_SPP_VALUES = ["4", "16", "32", "64"]