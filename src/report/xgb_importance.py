import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.report.funs._plot_style import PlotStyle
from src.report.funs._load_setup import load_setup


# -- CONFIG --
save = True


# -- LOAD DATA --
_, _, _, best_model, model = load_setup(task="clf")

# Load variable mapping
lookup_vars = pd.read_excel("data/dictionaries/lookup_vars_graph.xlsx")
var_mapping = dict(zip(lookup_vars["variable"], lookup_vars["graph_name"]))


# -- CREATE IMPORTANCE TABLE --
# Get weight scores
gain_scores = model.get_booster().get_score(importance_type="gain")
weight_scores = model.get_booster().get_score(importance_type="weight")

# Create dataframes with original and mapped names
# Gain
gain_df = pd.DataFrame([
    {
        "variable": k,  # original variable name
        "feature": var_mapping.get(k, k),  # mapped name used in plots
        "value": v,
    }
    for k, v in gain_scores.items()
])
gain_df["value"] = (gain_df["value"] / gain_df["value"].sum()) * 100
gain_df = gain_df.rename(columns={"value": "gain_score"})

# Weight
weight_df = pd.DataFrame([
    {"variable": k, "feature": var_mapping.get(k, k), "value": v}
    for k, v in weight_scores.items()
])
weight_df["value"] = (weight_df["value"] / weight_df["value"].sum()) * 100
weight_df = weight_df.rename(columns={"value": "weight_score"})

# Sort
gain_df = gain_df.sort_values("gain_score", ascending=False).reset_index(drop=True)
weight_df = weight_df.sort_values("weight_score", ascending=False).reset_index(drop=True)

# Select one from each table until 15 features are selected
selected = []
seen = set()

for g, w in zip(gain_df["variable"], weight_df["variable"]):
    for var in [w, g]:
        if var not in seen:
            selected.append(var)
            seen.add(var)
        if len(selected) == 10:
            break
    if len(selected) == 10:
        break

# Create DataFrame with selected features
# Select features
selected_df = pd.DataFrame({"variable": selected})

# Merge with gain and weight dataframes
selected_df = selected_df.merge(
    gain_df[["variable", "feature", "gain_score"]], on="variable", how="left"
).merge(weight_df[["variable", "weight_score"]], on="variable", how="left")


# -- CREATE PLOT --
# Sort by combined importance
selected_df["combined"] = selected_df["gain_score"] + selected_df["weight_score"]
selected_df = selected_df.sort_values("combined", ascending=True).reset_index(drop=True)

# Prepare values
y = np.arange(len(selected_df))
gain_vals = -selected_df["gain_score"]  # LEFT (negative side)
weight_vals = selected_df["weight_score"]  # RIGHT

# Plot
style = PlotStyle().apply()
fig, ax = plt.subplots(figsize=PlotStyle.figsize_from_pt(fraction=1, ratio=0.8))

# Bars
ax.barh(y, gain_vals, color=style.colors["low"], height=0.6, alpha=0.8)
ax.barh(y, weight_vals, color=style.colors["high"], height=0.6, alpha=0.8)

# Center line
ax.axvline(0, color=style.colors["text"], linewidth=0.5)

# Axis limits
ax.set_xlim(
    -selected_df["gain_score"].max() * 1.2,
    selected_df["weight_score"].max() * 1.2,
)

# Tick labels
ax.set_yticks(y)
ax.set_yticklabels(
    selected_df["feature"],
    color=style.colors["text"],
)
ax.set_xticklabels(
    [f"{abs(x):.0f}" for x in ax.get_xticks()],
    color=style.colors["text"],
)

# Axis label
ax.set_xlabel(
    "Relative Importance (%)", fontsize=style.base_font_size, color=style.colors["text"]
)
ax.set_yticklabels(
    selected_df["feature"],
    fontsize=style.base_font_size - 0.5,
    color=style.colors["text"],
)

# Apply base styling
style.style_axes(ax)

# Score labels
for i, row in selected_df.iterrows():
    ax.text(
        gain_vals[i] - 0.05,
        y[i],
        f"{row['gain_score']:.1f}%",
        ha="right",
        va="center",
        color=style.colors["text"],
        fontsize=style.base_font_size - 1.5,
    )
    ax.text(
        weight_vals[i] + 0.05,
        y[i],
        f"{row['weight_score']:.1f}%",
        ha="left",
        va="center",
        color=style.colors["text"],
        fontsize=style.base_font_size - 1.5,
    )

# Add category labels above bars
xlim = ax.get_xlim()
ylim = ax.get_ylim()
y_text = ylim[1] + 0.2

ax.text(
    xlim[0] / 2,
    y_text,
    "Gain",
    ha="center",
    va="bottom",
    color=style.colors["text"],
    fontsize=style.base_font_size - 0.5,
)
ax.text(
    xlim[1] / 2,
    y_text,
    "Weight",
    ha="center",
    va="bottom",
    color=style.colors["text"],
    fontsize=style.base_font_size - 0.5,
)

plt.tight_layout()

if save:
    plt.savefig("output/figs/xgb_importance_bf.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("output/figs/ppt/xgb_importance_bf.svg", bbox_inches="tight")
plt.show()
