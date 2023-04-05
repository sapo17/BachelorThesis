import numpy as np
import matplotlib.pyplot as plt

font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
}
plt.rc('font', **font)

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
TOL_DARKRED_COLOR = "#882255"

fig = plt.figure(figsize=(11, 3.5))
nrows = 1
ncols = 2
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0)
axs = gs.subplots(sharex="col") # real-world

# real-world
labels = [
    "Alginate (Green)",
    "Alginate (Blue)",
    "Birdy (Reconstructed)",
    "Birdy (Scaled Sphere)",
]
colors = [TOL_GREEN_COLOR, TOL_BLUE_COLOR, TOL_GOLD_COLOR, TOL_PINK_COLOR]
linestyles = ["solid", "solid", "solid", "solid"]
linewidths = [3, 3, 4, 4]

plots = [plot0, plot1, plot2, plot3]
for plot, color, linestyle, linewidth in zip(
    plots[:2], colors[:2], linestyles[:2], linewidths[:2]
):
    axs[0].plot(plot, color=color, linestyle=linestyle, linewidth=linewidth)
for plot, color, linestyle, linewidth in zip(
    plots[2:], colors[2:], linestyles[2:], linewidths[2:]
):
    axs[1].plot(plot, color=color, linestyle=linestyle, linewidth=linewidth)


axs[0].yaxis.set_major_locator(plt.MaxNLocator(5))
axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
axs[0].xaxis.set_major_locator(plt.MaxNLocator(5))
axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))

# real-world: legend
from matplotlib.lines import Line2D

custom_lines = [
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

import matplotlib as mpl
mpl.rcParams["legend.borderpad"] = 0.0
legend1 = fig.legend(
    custom_lines[:2],
    [
        "Alginate (Green)",
        "Alginate (Blue)",
    ],
    ncols=1,
    bbox_to_anchor=(0.565, 1.02),
    # bbox_to_anchor=(0.345, 0.45),
    fontsize=14
)
legend2 = fig.legend(
    custom_lines[2:],
    [
        "Birdy (Reconstructed)",
        "Birdy (Scaled Sphere)",
    ],
    ncols=1,
    bbox_to_anchor=(1.0, 1.02),
    fontsize=14
)

fig.supxlabel("Iteration")
fig.supylabel("Loss")

# real-world
fig.subplots_adjust(
    top=1.0,bottom=0.205,left=0.125,right=0.995,hspace=0.2,wspace=0.2
)
plt.savefig(
    "plot-real-world-v3.pdf", pad_inches=0.0, dpi=300, transparent=True
)
plt.show()
