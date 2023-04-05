import numpy as np
import matplotlib.pyplot as plt

font = {
        # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
}
plt.rc('font', **font)

## successfull attemps -- synthetic
plot0 = np.load(
    r"images\optimized\CESCG\updated\2023-03-20_12-51-32_iteration_97\loss_histroy.npy"
)
plot1 = np.load(
    r"images\optimized\CESCG\updated\2023-03-20_13-25-57_iteration_88\loss_histroy.npy"
)
plot2 = np.load(
    r"images\optimized\CESCG\updated\2023-03-24_14-06-47_iteration_99\loss_histroy.npy"
)
plot31 = np.load(
    r"images\optimized\CESCG\updated\2023-03-20_12-22-55_iteration_49\loss_histroy.npy"
)
plot32 = np.load(
    r"images\optimized\CESCG\updated\2023-03-20_12-22-55_iteration_49\2023-03-20_12-29-47_iteration_18\loss_histroy.npy"
)
plot3 = np.concatenate((plot31, plot32))

# ### unsuccessful attemps -- synthethic
plot4 = np.load(
    r"images\optimized\CESCG\updated\2023-03-24_15-14-36_iteration_99\loss_histroy.npy"
)
plot5 = np.load(
    r"images\optimized\CESCG\updated\2023-03-21_13-49-29_iteration_75\loss_histroy.npy"
)
plot6 = np.load(
    r"images\optimized\CESCG\updated\2023-03-24_10-58-50_iteration_88\loss_histroy.npy"
)
plot7 = np.load(
    r"images\optimized\CESCG\updated\2023-03-21_14-05-19_iteration_82\loss_histroy.npy"
)

import matplotlib.pyplot as plt
import numpy as np

# Color Blind friendly colors: https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
TOL_GOLD_COLOR = "#DDCC77"
TOL_BLUE_COLOR = "#88CCEE"
TOL_GREEN_COLOR = "#117733"
TOL_PINK_COLOR = "#CC6677"
TOL_DARKRED_COLOR = "#882255"

fig = plt.figure(figsize=(10.0, 3.0))
nrows = 1
ncols = 2
gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0)
axs = gs.subplots(sharex="col", sharey="row") # synthethic

# synthethic
labels = [
    "Bunny (Principled)",
    "Dragon (Principled)",
    "Bunny (Rough dielectric)",
    "Dragon (Rough dielectric)",
]
colors = [TOL_GREEN_COLOR, TOL_GOLD_COLOR, TOL_GREEN_COLOR, TOL_GOLD_COLOR]
linestyles = ["solid", "solid", "dashed", "dashed"]
linewidths = [3, 3, 4, 4]

# synthethic
plots = [plot0, plot1, plot2, plot3]
for plot, color, linestyle, linewidth in zip(plots, colors, linestyles, linewidths):
    axs[0].plot(plot, color=color, linestyle=linestyle, linewidth=linewidth)

axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
axs[0].xaxis.set_major_locator(plt.MaxNLocator(5))

# 2. column
plots = [plot4, plot5, plot6, plot7]
for plot, color, linestyle, linewidth in zip(plots, colors, linestyles, linewidths):
    axs[1].plot(
        plot, color=color, linestyle=linestyle, linewidth=linewidth
    )

axs[1].xaxis.set_major_locator(plt.MaxNLocator(5))

# synthethic: legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color="black", lw=2),
    Line2D([0], [0], color="black", lw=2, linestyle="dashed"),
    Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=TOL_GREEN_COLOR, markersize=10
    ),
    Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=TOL_GOLD_COLOR, markersize=10
    ),
]


legend = fig.legend(
    custom_lines, ["Principled BSDF", "Rough dielectric BSDF", "Bunny", "Dragon"], ncols = 1, fontsize=14, bbox_to_anchor=(0.55, 1.03)
)

# synthethic
axs[0].set_xlabel("Successful", loc="center")
axs[1].set_xlabel("Unsuccessful", loc="center")

fig.supxlabel("Iteration")
fig.supylabel("Loss")

#synthethic
fig.subplots_adjust(
top=1.0,bottom=0.29,left=0.11,right=0.995,
)

plt.savefig("plot-successful-unsuccessful-synthethic-v3.pdf", pad_inches=0.0, dpi=300, transparent=True)
plt.show()
