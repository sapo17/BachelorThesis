import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import glob
from PIL import Image
import matplotlib as mpl
from matplotlib import ticker


def get_paths(result_path: str):
    ABS_ERR_IMGS = result_path + r"\abs_err_imgs_it_*\*.png"
    OPT_IMGS = result_path + r"\opt_imgs_it_*\*.png"
    INIT_IMGS = result_path + r"\init_imgs\*.png"
    REF_IMGS = result_path + r"\ref_imgs\*.png"
    return INIT_IMGS, OPT_IMGS, REF_IMGS, ABS_ERR_IMGS


def getImagesFromRow(result_path):
    result = []
    for path in get_paths(result_path):
        img = Image.open(glob.glob(path)[0])
        img = resize_image(img, 256)
        result.append(img)
    return result


def resize_image(image: Image, length: int) -> Image:
    """
    Taken from: https://stackoverflow.com/questions/43512615/reshaping-rectangular-image-to-square, user: Titouan
    Resize an image to a square. Can make an image bigger to make it fit or smaller if it doesn't fit. It also crops
    part of the image.
    :param self:
    :param image: Image to resize.
    :param length: Width and height of the output image.
    :return: Return the resized image.
    """

    """
    Resizing strategy : 
     1) We resize the smallest side to the desired dimension (e.g. 1080)
     2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
    """
    if image.size[0] < image.size[1]:
        # The image is in portrait mode. Height is bigger than width.

        # This makes the width fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize(
            (length, int(image.size[1] * (length / image.size[0])))
        )

        # Amount of pixel to lose in total on the height of the image.
        required_loss = resized_image.size[1] - length

        # Crop the height of the image so as to keep the center part.
        resized_image = resized_image.crop(
            box=(
                0,
                required_loss / 2,
                length,
                resized_image.size[1] - required_loss / 2,
            )
        )

        # We now have a length*length pixels image.
        return resized_image
    else:
        # This image is in landscape mode or already squared. The width is bigger than the heihgt.

        # This makes the height fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize(
            (int(image.size[0] * (length / image.size[1])), length)
        )

        # Amount of pixel to lose in total on the width of the image.
        required_loss = resized_image.size[0] - length

        # Crop the width of the image so as to keep 1080 pixels of the center part.
        resized_image = resized_image.crop(
            box=(
                required_loss / 2,
                0,
                resized_image.size[0] - required_loss / 2,
                length,
            )
        )

        # We now have a length*length pixels image.
        return resized_image


UPDATED_DIR = r"images\optimized\CESCG\updated"

### successful: synthethic
# RESULT_0 = UPDATED_DIR + r"\2023-03-20_12-51-32_iteration_97"
# RESULT_1 = UPDATED_DIR + r"\2023-03-20_13-25-57_iteration_88"
# RESULT_2 = UPDATED_DIR + r"\2023-03-24_14-06-47_iteration_99"
# RESULT_3 = (
#     UPDATED_DIR
#     + r"\2023-03-20_12-22-55_iteration_49\2023-03-20_12-29-47_iteration_18"
# )

### unsuccessful: synthethic
# RESULT_0 = UPDATED_DIR + r"\2023-03-24_15-14-36_iteration_99"
# RESULT_1 = UPDATED_DIR + r"\2023-03-21_13-49-29_iteration_75"
# RESULT_2 = UPDATED_DIR + r"\2023-03-24_10-58-50_iteration_88"
# RESULT_3 = (
#     UPDATED_DIR
#     + r"\2023-03-21_14-05-19_iteration_82"
# )

### real-world
RESULT_0 = UPDATED_DIR + r"\2023-03-20_16-06-17_iteration_49"
RESULT_1 = (
    UPDATED_DIR
    + r"\2023-03-20_16-09-41_iteration_14\2023-03-20_16-14-04_iteration_46"
)
RESULT_2 = UPDATED_DIR + r"\2023-03-24_18-44-48_iteration_61"
RESULT_3 = UPDATED_DIR + r"\2023-03-20_18-18-45_iteration_99"

row0 = getImagesFromRow(RESULT_0)
row1 = getImagesFromRow(RESULT_1)
row2 = getImagesFromRow(RESULT_2)
row3 = getImagesFromRow(RESULT_3)
rows = row0 + row1 + row2 + row3

fig = plt.figure(figsize=(8.0, 8.0))
nrow_ncols = (4, 4)
grid = ImageGrid(
    fig,
    111,
    nrows_ncols=nrow_ncols,
    # axes_pad=0.01,
    share_all=True,
    cbar_mode="single",
    direction="row",
    cbar_size="2%",
    cbar_pad="0.5%",
)
for i, (ax, im) in enumerate(zip(grid, rows)):
    right_most_col = nrow_ncols[0] - 1
    if i % right_most_col == 0:
        img = ax.imshow(
            im, cmap="inferno", norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        )
    else:
        img = ax.imshow(im)

    if i == 0:
        ax.set_title("(a) Initial")
    elif i == 1:
        ax.set_title("(b) Optimized")
    elif i == 2:
        ax.set_title("(c) Reference")
    elif i == 3:
        ax.set_title("(d) Abs. Err.")

    ax.axis("off")

cbar = grid.cbar_axes[0].colorbar(img)  # uses last image
tick_font_size = 12
cbar.ax.tick_params(labelsize=tick_font_size)

fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=0.955)  # this

plt.savefig(
    "results_real_world.png",
    pad_inches=0.0,
    dpi=300,
    transparent=True,
)

plt.show()
