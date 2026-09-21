import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color="0.18", linestyle="--", linewidth=1.2,
           label="Reference"),
    Line2D([0], [0], color="#D55E00", linestyle="--", linewidth=1.2,
           label="Without time differentiation"),
    Line2D([0], [0], color="#0072B2", linestyle="-", linewidth=1.2,
           label="1st-order differentiation"),
]

fig = plt.figure(figsize=(5.5, 0.28))

fig.legend(
    handles=handles,
    loc="center",
    ncol=3,
    frameon=False,
    fontsize=8,
    handlelength=2.0,
    columnspacing=1.5,
    handletextpad=0.5,
)

fig.savefig(
    "derivative_ablation_legend.pdf",
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.close(fig)