import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from src.report.funs._plot_style import PlotStyle


# -- CONFIG --
save = True

# -- MAIN BOXPLOT AND KDE PLOT --
# Load features
features = pd.read_csv("data/interim/features.csv")
features = features[["eu_city_code", "eu_cycling"]]

eu = features.dropna(subset=["eu_cycling"])["eu_cycling"] * 100  # Convert to percentage

# Calculate median and mean
median_value = np.median(eu)
mean_value = np.mean(eu)
max_value = np.max(eu)
min_value = np.min(eu)

# Initialize style
style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=1.0, ratio=0.45)

# Create figure
fig, (ax_box, ax_kde) = plt.subplots(
    1, 2, gridspec_kw={"width_ratios": [1, 2.5]}, figsize=figsize
)

# Create boxplot
sns.boxplot(
    y=eu,
    ax=ax_box,
    color=style.colors["grey"],
    width=0.5,
    boxprops={
        "alpha": 0.4,
        "edgecolor": style.colors["grey"],
    },
    flierprops=dict(
        marker="o",
        markersize=1.5,
        alpha=0.4,
        linestyle="none",
        markeredgecolor=style.colors["grey"],
    ),
    whiskerprops=dict(color=style.colors["grey"], linewidth=1, alpha=0.4),
    capprops=dict(color=style.colors["grey"], linewidth=1, alpha=0.4),
    medianprops=dict(color=style.colors["grey"], linewidth=1, alpha=0.6),
)

# Create KDE
kde = stats.gaussian_kde(eu, bw_method=0.4)

x_grid = np.linspace(0, 50, 200)
kde_vals = kde.evaluate(x_grid) * 100

ax_kde.fill_between(x_grid, kde_vals, color=style.colors["grey"], alpha=0.4)
ax_kde.plot(x_grid, kde_vals, color=style.colors["grey"], linewidth=1, alpha=0.6)

# Add mean and median lines
ax_kde.axvline(
    median_value,
    color=style.colors["text"],
    linestyle=":",
    linewidth=1,
    alpha=0.6,
    label="Median",
)
ax_kde.axvline(
    mean_value,
    color=style.colors["text"],
    linestyle="--",
    linewidth=1,
    alpha=0.6,
    label="Mean",
)

# Legend
ax_kde.legend(frameon=False, loc="upper right", labelcolor=style.colors["text"])

# Thresholds at 5% and 15%
for thresh in [5, 15]:
    ax_kde.axvline(
        thresh,
        color=style.colors["text"],
        linestyle="-",
        linewidth=0.8,
        alpha=0.6,
        label="Thresholds" if thresh == 5 else None,  # Only label the first one
    )
ax_kde.legend(frameon=False, loc="upper right", labelcolor=style.colors["text"])

# Apply styling to both plots
style.style_axes(ax_box)
style.style_axes(ax_kde)

# Labels
ax_box.set_ylabel("Cycling Mode Share (%)")
ax_kde.set_xlabel("Cycling Mode Share (%)")
ax_kde.set_ylabel("Density")

# Set limits
ax_box.set_ylim(0, 53)
ax_kde.set_xlim(0, 50)
ax_kde.set_ylim(0, 9)

plt.tight_layout()

# Save
if save:
    fig.savefig("output/figs/mode_share_box_kde.png", bbox_inches="tight")
plt.show()
