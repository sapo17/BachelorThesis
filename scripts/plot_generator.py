import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant("cuda_ad_rgb")

### successfull attemps -- synthetic
# plot0 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-20_12-51-32_iteration_97\loss_histroy.npy"
# )
# plot1 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-20_13-25-57_iteration_88\loss_histroy.npy"
# )
# plot2 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-24_14-06-47_iteration_99\loss_histroy.npy"
# )
# plot31 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-20_12-22-55_iteration_49\loss_histroy.npy"
# )
# plot32 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-20_12-22-55_iteration_49\2023-03-20_12-29-47_iteration_18\loss_histroy.npy"
# )
# plot3 = np.concatenate((plot31, plot32))

# ### unsuccessful attemps -- synthethic
# plot4 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-24_15-14-36_iteration_99\loss_histroy.npy"
# )
# plot5 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-21_13-49-29_iteration_75\loss_histroy.npy"
# )
# plot6 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-24_10-58-50_iteration_88\loss_histroy.npy"
# )
# plot7 = np.load(
#     r"images\optimized\CESCG\updated\2023-03-21_14-05-19_iteration_82\loss_histroy.npy"
# )

### real world
UPDATED_DIR = r"images\optimized\CESCG\updated"
RESULT_01 = (
    UPDATED_DIR
    + r"\2023-03-20_16-06-17_iteration_49\2023-03-20_15-16-52_iteration_34\loss_histroy.npy"
)
RESULT_02 = UPDATED_DIR + r"\2023-03-20_16-06-17_iteration_49\loss_histroy.npy"
RESULT_11 = UPDATED_DIR + r"\2023-03-20_16-09-41_iteration_14\loss_histroy.npy"
RESULT_12 = (
    UPDATED_DIR
    + r"\2023-03-20_16-09-41_iteration_14\2023-03-20_16-14-04_iteration_46\loss_histroy.npy"
)

RESULT_2 = UPDATED_DIR + r"\2023-03-24_18-44-48_iteration_61\loss_histroy.npy"
RESULT_3 = UPDATED_DIR + r"\2023-03-20_18-18-45_iteration_99\loss_histroy.npy"

plot01 = np.load(RESULT_01)
plot02 = np.load(RESULT_02)
plot0 = np.concatenate((plot01, plot02))

plot11 = np.load(RESULT_11)
plot12 = np.load(RESULT_12)
plot1 = np.concatenate((plot11, plot12))

plot2 = np.load(RESULT_2)
plot3 = np.load(RESULT_3)

import matplotlib.pyplot as plt
import numpy as np

# Color Blind friendly colors: https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
TOL_GOLD_COLOR = "#DDCC77"
TOL_BLUE_COLOR = "#88CCEE"
TOL_GREEN_COLOR = "#117733"
TOL_PINK_COLOR = "#CC6677"

# fig = plt.figure(figsize=(8.0, 6.0)) # synthethic
fig = plt.figure(figsize=(10.0, 6.0))  # real-world
nrows = 4
# ncols = 1 # synthethic
ncols = 1  # real-world
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0)
axs = gs.subplots(sharex="col", sharey="row")

# synthethic
# labels = [
#     "Bunny (Principled)",
#     "Dragon (Principled)",
#     "Bunny (Rough dielectric)",
#     "Dragon (Rough dielectric)",
# ]
# colors = [TOL_BLUE_COLOR, TOL_GOLD_COLOR, TOL_BLUE_COLOR, TOL_GOLD_COLOR]
# linestyles = ["solid", "solid", "dashed", "dashed"]

# real-world
labels = [
    "Alginate (Green)",
    "Alginate (Blue)",
    "Birdy (Reconstructed)",
    "Birdy (Scaled Sphere)",
]
colors = [TOL_GREEN_COLOR, TOL_BLUE_COLOR, TOL_GOLD_COLOR, TOL_PINK_COLOR]
linestyles = ["solid", "solid", "solid", "solid"]

plots = [plot0, plot1, plot2, plot3]
for rowIdx, plot, color, linestyle in zip(
    range(nrows), plots, colors, linestyles
):
    axs[rowIdx].plot(plot, color=color, linestyle=linestyle, linewidth=2)
    axs[rowIdx].yaxis.set_major_locator(plt.MaxNLocator(3))
    axs[rowIdx].xaxis.set_major_locator(plt.MaxNLocator(5))


# synthethic, 2. column
# plots = [plot4, plot5, plot6, plot7]
# for rowIdx, plot, color, linestyle in zip(
#     range(nrows), plots, colors, linestyles
# ):
#     axs[rowIdx, 1].plot(
#         plot, color=color, linestyle=linestyle, linewidth=2
#     )
#     axs[rowIdx, 1].xaxis.set_major_locator(plt.MaxNLocator(5))

# synthethic: legend
# from matplotlib.lines import Line2D
# custom_lines = [
#     Line2D([0], [0], color="black", lw=2),
#     Line2D([0], [0], color="black", lw=2, linestyle="dashed"),
#     Line2D(
#         [0], [0], marker="o", color="w", markerfacecolor=TOL_BLUE_COLOR, markersize=10
#     ),
#     Line2D(
#         [0], [0], marker="o", color="w", markerfacecolor=TOL_GOLD_COLOR, markersize=10
#     ),
# ]
# fig.legend(
#     custom_lines, ["Principled BSDF", "Rough dielectric BSDF", "Bunny", "Dragon"], ncols = 4, loc="upper center"
# )

# real-world: legend
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="black", lw=2),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=TOL_GREEN_COLOR,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=TOL_BLUE_COLOR,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=TOL_GOLD_COLOR,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=TOL_PINK_COLOR,
        markersize=10,
    ),
]

fig.legend(
    custom_lines,
    [
        "Principled BSDF",
        "Alginate (Green)",
        "Alginate (Blue)",
        "Birdy (Reconstructed)",
        "Birdy (Scaled Sphere)",
    ],
    ncols=5,
    loc="upper center",
)

# synthethic
axs[0].set_title("Successful")

# real-world
# axs[0, 0].set_title("Successful")
# axs[0, 1].set_title("Unsuccessful")

fig.supxlabel("Iteration")
fig.supylabel("Loss")
# synthetic
# fig.subplots_adjust(
# top=0.9,bottom=0.085,left=0.105,right=1.0,
# )
# real-world
fig.subplots_adjust(
    top=0.9,
    bottom=0.085,
    left=0.085,
    right=1.0,
)  # this
# plt.savefig("plot-successful-unsuccessful-synthethic.png", pad_inches=0.0, dpi=300, transparent=True)
plt.savefig("plot-real-world.png", pad_inches=0.0, dpi=300, transparent=True)
plt.show()
