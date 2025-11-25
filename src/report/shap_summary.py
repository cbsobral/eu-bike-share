import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from src.model.funs.classifiers import create_target_categories
from src.report._plot_style import PlotStyle
from src.report._load_setup import load_setup


# -- CONFIG --
save = True


# -- LOAD AND PREPARE DATA --
# Load data and model
data_clf, _, _, _, best_model = load_setup(task="clf")
data_clf = data_clf.dropna(subset=["eu_cycling"])
X = data_clf.drop(columns=["eu_cycling", "eu_city_code", "eu_city"])
y = data_clf["eu_cycling"]

# Load variable name mapping
lookup_vars = pd.read_excel("data/dictionaries/lookup_vars_graph.xlsx")
var_map = dict(zip(lookup_vars["variable"], lookup_vars["graph_name"]))
feat_names = [var_map.get(c, c) for c in X.columns]


# -- SHAP VALUES -
_, y_enc, le = create_target_categories(y)
n_classes = len(le.classes_)

# SHAP from full model
expl_clf = shap.TreeExplainer(best_model)
shap_clf = expl_clf.shap_values(X)

# Select class
shap_high = shap_clf[:, :, 0]  # alphabetically -- 0 high, 1 low

# Label columns
X.columns = feat_names

# Select features based on importance across classes
abs_shap = np.abs(shap_clf)
overall_importance = abs_shap.mean(axis=0).mean(axis=1)
top_indices = np.argsort(overall_importance)[::-1][:12]

X_top = X.iloc[:, top_indices].copy()
shap_high_top = shap_high[:, top_indices]


# -- PLOT --
style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=1.0, ratio=0.55)
plt.figure(figsize=figsize)

custom_cmap = LinearSegmentedColormap.from_list(
    "custom", [style.colors["low"], style.colors["mid"], style.colors["high"]]
)

shap.summary_plot(
    shap_high_top,
    X_top,
    plot_type="dot",
    show=False,
    cmap=custom_cmap,
    max_display=12,
    sort=True,
)

plt.gcf().set_size_inches(figsize)
for coll in plt.gca().collections:
    if hasattr(coll, "set_sizes"):
        coll.set_sizes([6])

plt.xlabel(
    "Effect on Predicted Probability of High Cycling Class (SHAP Value)", fontsize=9
)
plt.gca().tick_params(axis="y", labelsize=9)
plt.gca().tick_params(axis="x", labelsize=8)
style.style_axes(plt.gca())

cbar = plt.gcf().axes[-1]
cbar.set_ylabel("Feature Value", fontsize=8)
cbar.tick_params(labelsize=8)
cbar.yaxis.set_label_coords(3.0, 0.5)

if save:
    plt.savefig("output/figs/shap_summary_high.png", bbox_inches="tight", dpi=600)

plt.show()
