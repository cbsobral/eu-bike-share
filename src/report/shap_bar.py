import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy import stats
from src.report._plot_style import PlotStyle
from src.report._load_setup import load_setup


# -- CONFIG --
save = True
city_codes = ["BE012C", "BE015C"]  # Mechelen, Verviers
city_classes = {"BE012C": "High", "BE015C": "Low"}
top_k = 10


# -- LOAD AND PREPARE DATA --
conformal = pd.read_csv("models/clf_conformal_predictions.csv")
data_clf, _, _, _, best_model = load_setup(task="clf")
lookup_vars = pd.read_excel("data/dictionaries/lookup_vars_graph.xlsx")

# Create variable name mapping
var_map = dict(zip(lookup_vars["variable"], lookup_vars["graph_name"]))
orig_features = [
    c for c in data_clf.columns if c not in (["eu_cycling", "eu_city_code", "eu_city"])
]


# -- CREATE SIDE BY SIDE PLOTS --
style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=1.5, ratio=0.3)
fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

for i, city_code in enumerate(city_codes):
    # Select city
    city_row = data_clf.loc[data_clf["eu_city_code"] == city_code].copy()
    x_city = city_row[orig_features].reindex(columns=orig_features)

    # Predict
    y_proba_city = best_model.predict_proba(x_city)[0]
    y_pred_city = best_model.predict(x_city)[0]

    # Explainer for SHAP values
    explainer = shap.TreeExplainer(best_model)
    shap_raw = explainer.shap_values(x_city)

    if isinstance(shap_raw, list):
        shap_all = np.stack(shap_raw, axis=-1)
        expected_vals = explainer.expected_value
    else:
        shap_all = shap_raw
        expected_vals = explainer.expected_value

    pred_class_idx = int(np.argmax(y_proba_city))
    shap_vec = shap_all[0, :, pred_class_idx]
    base_val = (
        expected_vals[pred_class_idx]
        if isinstance(expected_vals, (list, np.ndarray))
        else expected_vals
    )

    # Build explanation object
    display_names = [var_map.get(f, f) for f in orig_features]
    data_values = x_city.iloc[0].values

    # Remove NaNs and get top k features
    mask_nonan = ~pd.isna(data_values)
    valid_indices = np.where(mask_nonan)[0]
    valid_shap_values = shap_vec[mask_nonan]
    top_idx = np.argsort(np.abs(valid_shap_values))[::-1][:top_k]

    # Normalize values to percentiles
    percentile_values = []
    for j in top_idx:
        feature_col = orig_features[valid_indices[j]]
        percentile = stats.percentileofscore(
            data_clf[feature_col].dropna(), data_values[valid_indices[j]]
        )
        p_int = int(min(max(round(percentile), 0), 99))  # Clip at 99 to avoid 100
        percentile_values.append(f"p{p_int}")

    shap_expl_final = shap.Explanation(
        values=valid_shap_values[top_idx],
        base_values=base_val,
        data=np.array(percentile_values),
        feature_names=[display_names[valid_indices[j]] for j in top_idx],
    )

    # Plot on subplot
    ax = axes[i]
    shap.plots.bar(shap_expl_final, max_display=top_k, show=False, ax=ax)
    # Make the zero line (middle bar) the spine color
    ax.axvline(x=0, color=style.colors["spines"], linewidth=0.8, zorder=1)
    style.style_axes(ax)

    # Apply custom colors
    bars = ax.patches
    for j, bar in enumerate(bars):
        if shap_expl_final.values[j] > 0:
            bar.set_color(style.colors["teal"])
        else:
            bar.set_color(style.colors["grey"])
        bar.set_alpha(0.8)
        bar.set_edgecolor("none")

    # Apply consistent font sizes and styling
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(f"SHAP Value for {city_classes[city_code]} Class", fontsize=8)

    # Get city name for title
    city_name = city_row["eu_city"].values[0]
    ax.set_title(f"{city_name}", fontsize=9)

    # Apply text formatting to SHAP value labels on bars
    for text in ax.texts:
        text.set_fontsize(7)
        text.set_color(style.colors["text"])

    # Override the gray tick labels to be even lighter
    tick_labels = ax.yaxis.get_majorticklabels()
    for k in range(top_k):
        tick_labels[k].set_color("#999999")


plt.tight_layout()

if save:
    plt.savefig("output/shap_bar.png", bbox_inches="tight", dpi=600)

plt.show()
