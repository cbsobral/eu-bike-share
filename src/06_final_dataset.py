"""06_final_dataset.py
writes the published dataset of observed and predicted cycling for all 864 cities
"""

import joblib
import numpy as np
import pandas as pd

from utils import OUTPUT_DIR, TARGET, load_model_data


# -- BUILD DATASET --
# Observed shares and predicted classes
predictions = pd.read_csv(OUTPUT_DIR / "predictions.csv")
dataset = predictions[
    [
        "eu_city_code",
        "eu_city",
        "eu_country",
        "predicted",
        TARGET,
        "predicted_class",
        "prediction_set",
    ]
].rename(
    columns={
        "eu_city_code": "city_code",
        "eu_city": "city",
        "eu_country": "country",
        "predicted_class": "pred_class",
    }
)
dataset[TARGET] = dataset[TARGET].round(
    3
)  # keeps shares at 0.05 and 0.15 in their class

# Predicted shares
data, _, features, _, _, _ = load_model_data()
model = joblib.load(OUTPUT_DIR / "regression_model.joblib")
dataset.insert(6, "pred_reg", np.nan)
dataset.loc[dataset["predicted"], "pred_reg"] = np.clip(
    model.predict(data.loc[dataset["predicted"], features]), 0, None
)
dataset["pred_reg"] = dataset["pred_reg"].round(2)


# -- SAVE --
dataset.to_csv("cycling_mode_share.csv", index=False)
