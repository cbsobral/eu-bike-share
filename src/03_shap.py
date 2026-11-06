"""03_shap.py
ranks the selected predictors by mean absolute shap value across classes
"""

import joblib
import numpy as np
import pandas as pd
import shap

from utils import OUTPUT_DIR, load_model_data


# -- EXPLAIN --
_, labeled, features, _, _, _ = load_model_data()
model = joblib.load(OUTPUT_DIR / "classification_model.joblib")
shap_values = shap.TreeExplainer(model).shap_values(labeled[features])


# -- RANK AND SAVE --
importance = pd.read_csv(
    OUTPUT_DIR / "selected_features.csv", usecols=["variable", "label"]
)
importance["mean_abs_shap"] = np.abs(shap_values).mean(axis=(0, 2))
importance = importance.sort_values("mean_abs_shap", ascending=False)
importance.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
