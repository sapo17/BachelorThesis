import math
from pathlib import Path
import re
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def get_paths_2(result_path: str):
    ABS_ERR_IMGS = result_path + r"\abs_err_imgs_it_*\*.png"
    INIT_IMGS = result_path + r"\init_imgs\*.png"
    REF_IMGS = result_path + r"\ref_imgs\*.png"
    return INIT_IMGS, REF_IMGS, ABS_ERR_IMGS


def getImagesFromRow(result_path):
    result = []
    for path in get_paths_2(result_path):
        img = Image.open(glob.glob(path)[0])
        img = resize_image(img, 256)
        result.append(img)
    return result

def get_paths(result_path: str):
    return glob.glob(result_path + r"\sensor_*")


def get_images_path(path: str):
    return glob.glob(path + r"\\*.png")


def get_path_to_images(path):
    return [
        Image.open(img_path)
        for p in get_paths(path)
        for img_path in get_images_path(p)
    ]


file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    # taken from: https://stackoverflow.com/questions/62941378/how-to-sort-glob-glob-numerically
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


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
    
def createFigure(rows, it, diffRenderStackedPath):
    # fig = plt.figure(figsize=(6, 8))
    fig = plt.figure(figsize=(6, 4))
    nrow_ncols = (2, 3)
    # nrow_ncols = (4, 3)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=nrow_ncols,
        # axes_pad=0.01,
        share_all=True,
        direction="row",
    )
    for i, (ax, im) in enumerate(zip(grid, rows)):
        ax.imshow(im)

        if i == 0:
            ax.set_title("(a) Initial")
        elif i == 1:
            ax.set_title(f"(b) Diff. render it. {it}")
        elif i == 2:
            ax.set_title("(c) Reference")

        ax.axis("off")

    # fig.subplots_adjust(top=0.968,bottom=0.0,left=0.0,right=1.0)  # this
    fig.subplots_adjust(top=0.932,bottom=0.0,left=0.0,right=1.0,hspace=0.2,wspace=0.2)  # this
    # plt.tight_layout()

    fileName = diffRenderStackedPath + f"\\{it}.png"
    plt.savefig(
        fileName,
        pad_inches=0.0,
        # dpi=300,
    )

    # plt.show()

### NOTE: Data used here is zipped and not included in the repo.

# synthetic
# DIR = r"images\optimized\CESCG\updated\video-submission-data\2023-04-07_17-56-29_iteration_99" # principled dragon
# DIR = r"images\optimized\CESCG\updated\video-submission-data\2023-04-07_19-28-10_iteration_49" # rough dielectric dragon
# DIR_2 = DIR + r"\2023-04-07_19-36-22_iteration_49" # rough dielectric dragon

# real-world
DIR = r"images\optimized\CESCG\updated\video-submission-data\2023-04-08_10-55-36_iteration_49" # blue alginate
DIR_2 = DIR + r"\2023-04-08_11-02-49_iteration_49" # blue alginate-2
DIR2 = r"images\optimized\CESCG\updated\video-submission-data\2023-04-08_11-08-31_iteration_49" # green alginate, status: seems deleted 0.0 (TODO but reproducable)
DIR_21 = DIR2 + r"\2023-04-08_11-14-09_iteration_49" # green alginate-2, status: seems deleted 0.0 (TODO but reproducable)
DIFF_RENDER_DIR = DIR + r"\diff_render_history"
DIFF_RENDER_DIR_2 = DIR_2 + r"\diff_render_history"
DIFF_RENDER_DIR2 = DIR2 + r"\diff_render_history"
DIFF_RENDER_DIR_21 = DIR_21 + r"\diff_render_history"
REF_IMGS = DIR + r"\ref_imgs\*.png"

init0 = resize_image(Image.open(glob.glob(DIR + r"\init_imgs\*_s0.png")[0]), 256)
init1 = resize_image(Image.open(glob.glob(DIR2 + r"\init_imgs\*_s0.png")[0]), 256)
# init0 = resize_image(Image.open(glob.glob(DIR + r"\init_imgs\*_s0.png")[0]), 256)
# init1 = resize_image(Image.open(glob.glob(DIR + r"\init_imgs\*_s1.png")[0]), 256)
# init2 = resize_image(Image.open(glob.glob(DIR + r"\init_imgs\*_s2.png")[0]), 256)
# init3 = resize_image(Image.open(glob.glob(DIR + r"\init_imgs\*_s3.png")[0]), 256)

ref0 = resize_image(Image.open(glob.glob(DIR + r"\ref_imgs\*_s0.png")[0]), 256)
ref1 = resize_image(Image.open(glob.glob(DIR2 + r"\ref_imgs\*_s0.png")[0]), 256)
# ref0 = resize_image(Image.open(glob.glob(DIR + r"\ref_imgs\*_s0.png")[0]), 256)
# ref1 = resize_image(Image.open(glob.glob(DIR + r"\ref_imgs\*_s1.png")[0]), 256)
# ref2 = resize_image(Image.open(glob.glob(DIR + r"\ref_imgs\*_s2.png")[0]), 256)
# ref3 = resize_image(Image.open(glob.glob(DIR + r"\ref_imgs\*_s3.png")[0]), 256)

imgs00 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR)[0]), key=get_order)
imgs01 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_2)[0]), key=get_order)
imgs0 = imgs00 + imgs01
imgs10 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR2)[0]), key=get_order)
imgs11 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_21)[0]), key=get_order)
imgs1 = imgs10 + imgs11
# imgs00 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR)[0]), key=get_order)
# imgs01 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_2)[0]), key=get_order)
# imgs0 = imgs00 + imgs01
# imgs10 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR)[1]), key=get_order)
# imgs11 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_2)[1]), key=get_order)
# imgs1 = imgs10 + imgs11
# imgs20 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR)[2]), key=get_order)
# imgs21 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_2)[2]), key=get_order)
# imgs2 = imgs20 + imgs21
# imgs30 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR)[3]), key=get_order)
# imgs31 = sorted(get_images_path(get_paths(DIFF_RENDER_DIR_2)[3]), key=get_order)
# imgs3 = imgs30 + imgs31

imgs0 = [resize_image(Image.open(img_path), 256) for img_path in imgs0]
imgs1 = [resize_image(Image.open(img_path), 256) for img_path in imgs1]
# imgs0 = [resize_image(Image.open(img_path), 256) for img_path in imgs0]
# imgs1 = [resize_image(Image.open(img_path), 256) for img_path in imgs1]
# imgs2 = [resize_image(Image.open(img_path), 256) for img_path in imgs2]
# imgs3 = [resize_image(Image.open(img_path), 256) for img_path in imgs3]


diffRenderStackedPath = DIFF_RENDER_DIR + "\\images_stacked_2"
Path(diffRenderStackedPath).mkdir(parents=True, exist_ok=True)

# for it, (im0, im1, im2, im3) in enumerate(zip(imgs0, imgs1, imgs2, imgs3)):
for it, (im0, im1) in enumerate(zip(imgs0, imgs1)):
    row0 = [init0, im0, ref0]
    row1 = [init1, im1, ref1]
    # row0 = [init0, im0, ref0]
    # row1 = [init1, im1, ref1]
    # row2 = [init2, im2, ref2]
    # row3 = [init3, im3, ref3]
    # rows = row0 + row1 + row2 + row3
    rows = row0 + row1
    createFigure(rows, it, diffRenderStackedPath)

