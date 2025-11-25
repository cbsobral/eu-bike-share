import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.report.funs._plot_style import PlotStyle
from src.report.funs._load_setup import load_setup


# -- CONFIG --
save = True


# -- LOAD DATA --
data, predictions, final, info, _ = load_setup(task="clf")
cf = classification_report(
    predictions["y_true"], predictions["y_pred"], labels=["low", "moderate", "high"]
)


# -- CONFUSION MATRIX --
style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=0.8, ratio=0.75)

cm = confusion_matrix(
    predictions["y_true"], predictions["y_pred"], labels=["low", "moderate", "high"]
)

# Define category labels
class_labels = ["Low", "Moderate", "High"]

# Define a custom colormap with transparency
custom_cmap = sns.light_palette(style.colors["grey"], as_cmap=True)

fig, ax = plt.subplots(figsize=figsize)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=custom_cmap,
    alpha=0.85,
    xticklabels=class_labels,
    yticklabels=class_labels,
    linewidths=0.5,
    linecolor=style.colors["grid"],
    annot_kws={"fontsize": 8, "color": style.colors["text"]},
    cbar_kws={"shrink": 0.8},
    ax=ax,
)

# Emphasize the darkest (max value) cell
max_val = cm.max()
max_pos = np.argwhere(cm == max_val)[0]  # (row, col)
ax.text(
    max_pos[1] + 0.5,
    max_pos[0] + 0.5,
    str(max_val),
    ha="center",
    va="center",
    color="white",
    fontsize=8,
    fontweight="bold",
    family="sans-serif",
)

# Color
cbar = ax.collections[0].colorbar
for label in cbar.ax.get_yticklabels():
    label.set_family("sans-serif")

# Remove grid lines
ax.grid(False)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

# Save
if save:
    fig.savefig("output/figs/conf_matrix.pdf", bbox_inches="tight", dpi=600)
    fig.savefig("output/figs/ppt/conf_matrix.svg", bbox_inches="tight", dpi=600)

plt.show()
