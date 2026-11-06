"""04_uncertainty.py
predicts cycling levels for unobserved cities with 90% conformal prediction sets
"""

import joblib
import numpy as np
import pandas as pd
from mapie.classification import CrossConformalClassifier
from sklearn.base import clone

from utils import OUTPUT_DIR, TARGET, get_cv_strategy, load_model_data


# -- LOAD AND PREPARE --
data, labeled, features, _, y_encoded, encoder = load_model_data()
unlabeled = data.loc[data[TARGET].isna()]
model = joblib.load(OUTPUT_DIR / "classification_model.joblib")


# -- PREDICT UNOBSERVED CITIES --
mapie_model = CrossConformalClassifier(
    estimator=clone(model),
    cv=get_cv_strategy(),
    confidence_level=0.9,
    conformity_score="lac",
)
mapie_model.fit_conformalize(labeled[features], y_encoded)
y_pred, y_pred_sets = mapie_model.predict_set(
    unlabeled[features],
    conformity_score_params={"include_last_label": False},
    agg_scores="crossval",
)

# Prediction sets ordered by probability
probabilities = model.predict_proba(unlabeled[features])
prediction_sets = []
for row_set, row_proba in zip(y_pred_sets[:, :, 0], probabilities):
    members = np.where(row_set)[0]
    ordered = members[np.argsort(row_proba[members])[::-1]]
    prediction_sets.append(str(encoder.inverse_transform(ordered).tolist()))


# -- COMBINE AND SAVE --
cities = pd.read_csv("data/lookup_cities.csv", usecols=["eu_city_code", "eu_country"])
predictions = data[["eu_city_code", "eu_city", TARGET]].merge(
    cities, on="eu_city_code", how="left"
)
predictions["predicted"] = data[TARGET].isna()
predictions.loc[predictions["predicted"], "predicted_class"] = (
    encoder.inverse_transform(y_pred)
)
predictions.loc[predictions["predicted"], "prediction_set"] = prediction_sets
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
