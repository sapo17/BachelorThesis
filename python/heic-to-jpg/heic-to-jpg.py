import os
from PIL import Image
import pillow_heif

def heic_to_png(inputPath, inputDirs, outputPath):
    for item in inputDirs:
        itemPath = os.path.join(inputPath, item)
        itemOutputPath = os.path.join(outputPath, item)
        if os.path.isfile(itemPath):
            heif_file = pillow_heif.read_heif(itemPath)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            f, e = os.path.splitext(itemOutputPath)
            image.save(f + ".png", format("png"))


inputPath = 'docs/091122-sample-images'
outputPath = 'docs/091122-sample-images-png'
input_dirs = os.listdir(inputPath)

heic_to_png(inputPath, input_dirs, outputPath)