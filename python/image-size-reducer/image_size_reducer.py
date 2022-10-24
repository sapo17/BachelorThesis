from PIL import Image, ImageOps
import os
import sys

# [Method *resize()* is taken from: Stackoverflow - Python/PIL Resize all images in a folder](https://stackoverflow.com/questions/21517879/python-pil-resize-all-images-in-a-folder)
# Method modified to provide the needs


def resize(inputPath, inputDirs, outputPath, sizeTuple=(1024, 1024), quality=50):
    for item in inputDirs:
        itemPath = os.path.join(inputPath, item)
        itemOutputPath = os.path.join(outputPath, item)
        if os.path.isfile(itemPath):
            im = Image.open(itemPath)

            # handle rotation on save, https://stackoverflow.com/questions/4228530/pil-thumbnail-is-rotating-my-image
            im = ImageOps.exif_transpose(im)
            f, e = os.path.splitext(itemOutputPath)
            imResize = im.resize(sizeTuple)
            imResize.save(f + '.jpg', 'JPEG', quality=quality)


inputPath = 'instant-ngp/data/nerf/pizza/orig'
outputPath = 'instant-ngp/data/nerf/pizza/resized'
input_dirs = os.listdir(inputPath)

resize(inputPath, input_dirs, outputPath)
