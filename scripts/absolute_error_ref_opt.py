from PIL import Image, ImageChops, ImageOps
import matplotlib.pyplot as plt
import numpy as np

def normalized_absolute_error(ref: Image, opt: Image):
    diff = ImageChops.difference(ref, opt)
    diff = ImageOps.grayscale(diff)
    diff_np = np.array(diff, dtype="float64")
    diff_np *= 1.0 / diff_np.max()
    return diff_np

def save_as_plot(diff_np: np.array, filename: str):
    plt.imshow(diff_np, cmap="inferno")
    plt.colorbar(); plt.xticks([]); plt.yticks([]);
    plt.xlabel(r"Absolute Error: |$y_{ref} - y_{opt}$|", size=14)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

ref = Image.open("images\\references\\1673691134903_IMG_9822.jpg")
ref = ref.resize((256, 256))
opt = Image.open("images\optimized\green-circular\optimized.png")
diff_np = normalized_absolute_error(ref, opt)
save_as_plot(diff_np, "images\optimized\\green-circular\\absolute_error_ref_best_opt.png")