import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.report.funs._plot_style import PlotStyle
from src.report.funs._load_setup import load_setup


# -- CONFIG --
save = True


# -- LOAD DATA --
data, predictions, _, best_model, _ = load_setup(task="reg")


# -- CREATE RESIDUAL PLOT --
style = PlotStyle().apply()

# Filter only positive values of y_pred in predictions table
preds = predictions[predictions["y_pred"] > 0].copy()

# Calculate residuals
preds["residuals"] = preds["y_true"] - preds["y_pred"]

style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=1.0, ratio=0.45)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

# Custom colormap for true vs predicted
custom_cmap = LinearSegmentedColormap.from_list(
    "custom", [style.colors["low"], style.colors["mid"], style.colors["high"]]
)

# Predicted vs True plot
ax1.scatter(
    preds["y_pred"],
    preds["y_true"],
    alpha=0.9,
    s=9,
    c=preds["y_true"],
    cmap=custom_cmap,
    edgecolor="none",
)

# Add perfect prediction line
max_val = max(preds["y_true"].max(), preds["y_pred"].max())
ax1.plot([0, max_val], [0, max_val], "--", color=style.colors["text"], alpha=0.5)

# Add vertical bin boundaries
bin_boundaries = [0.05, 0.15]
for boundary in bin_boundaries:
    ax1.axvline(
        x=boundary,
        color=style.colors["text"],
        linewidth=0.8,
        linestyle=":",
    )

ax1.set_xlabel("Predicted Cycling Mode Share")
ax1.set_ylabel("True Cycling Mode Share")

# Residuals plot
ax2.scatter(
    preds["y_pred"],
    preds["residuals"],
    alpha=0.9,
    s=9,
    c=preds["y_pred"],
    cmap=custom_cmap,
    edgecolor="none",
)

# Add zero line and bin boundaries
ax2.axhline(y=0, color=style.colors["text"], linestyle="--", alpha=0.5)
for boundary in bin_boundaries:
    ax2.axvline(x=boundary, color=style.colors["text"], linestyle=":", linewidth=0.8)

ax2.set_xlabel("Predicted Cycling Mode Share")
ax2.set_ylabel("Residuals (True - Predicted)")

# Standardize x-axis ticks for both plots
xticks = np.arange(0, max_val + 0.05, 0.1)  # Every 5 percentage points
for ax in [ax1, ax2]:
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.2f}" for x in xticks])
    style.style_axes(ax)

# Adjust layout
plt.tight_layout()

# Save
if save:
    plt.savefig("output/figs/xgb_regression.png", bbox_inches="tight", dpi=600)
plt.show()
