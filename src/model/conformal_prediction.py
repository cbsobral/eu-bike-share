import numpy as np
import pandas as pd
from mapie.classification import CrossConformalClassifier
from sklearn.base import clone
from src.model.funs.classifiers import (
    prepare_features,
    create_target_categories,
    get_cv_strategy,
)
from src.report.funs._load_setup import load_setup


# -- CONFIG --
target = "eu_cycling"
bin_option = "simple"
confidence_level = 0.9
cv = get_cv_strategy()


# -- LOAD AND PREPARE DATA --
data, _, final, best_model, model = load_setup(target, task="clf")

# Split data into labeled and unlabeled sets
labeled_mask = data[target].notna()
data_labeled = data[labeled_mask].copy()
data_unlabeled = data[~labeled_mask].copy()

# Prepare features
X_labeled, y_labeled = prepare_features(data_labeled, target)
y_cat, y_encoded, le = create_target_categories(y_labeled)

X_unlabeled = prepare_features(data_unlabeled, target)[0]

city_codes_unlabeled = data_unlabeled["eu_city_code"].values
city_names_unlabeled = data_unlabeled["eu_city"].values


# Initialize CrossConformalClassifier
mapie_clf = CrossConformalClassifier(
    estimator=clone(model),
    cv=cv,
    confidence_level=confidence_level,
    conformity_score="lac",
)

# Fit and conformalize on labeled data
mapie_clf.fit_conformalize(X_labeled, y_encoded)

# Fit a separate model for probabilities
prob_model = model
prob_model.fit(X_labeled, y_encoded)

# Get probabilities and predictions
proba = prob_model.predict_proba(X_unlabeled)
y_pred_prob = prob_model.predict(X_unlabeled)

# Predict on unlabeled cities
y_pred, y_pred_sets = mapie_clf.predict_set(
    X_unlabeled,
    conformity_score_params={"include_last_label": False},
    agg_scores="crossval",
)

# Assert check: main class should be the same
assert np.array_equal(y_pred, y_pred_prob), "Different predictions from Mapie and model"


# -- PROCESS RESULTS --
# Convert back to original labels
y_pred_original = le.inverse_transform(y_pred)

# Process prediction sets - ORDER BY PROBABILITY
prediction_sets_original = []
prediction_set_sizes = []

for i in range(len(y_pred_sets)):
    pred_set = y_pred_sets[i, :, 0] if len(y_pred_sets.shape) == 3 else y_pred_sets[i]
    predicted_classes_idx = np.where(pred_set)[0]

    # Get probabilities for the predicted classes
    class_probs = proba[i, predicted_classes_idx]

    # Sort by probability (descending - highest first)
    sorted_indices = np.argsort(class_probs)[::-1]
    sorted_class_indices = predicted_classes_idx[sorted_indices]

    # Convert to original labels
    predicted_classes_sorted = le.inverse_transform(sorted_class_indices)

    prediction_sets_original.append(predicted_classes_sorted.tolist())
    prediction_set_sizes.append(len(predicted_classes_sorted))

# Create results DataFrame
conformal_preds = pd.DataFrame({
    "eu_city_code": city_codes_unlabeled,
    "eu_city": city_names_unlabeled,
    "predicted_class": y_pred_original,
    "prediction_set": prediction_sets_original,
    "prediction_set_size": prediction_set_sizes,
})

# Add probabilities
class_labels = le.classes_
proba_df = pd.DataFrame(proba, columns=[f"proba_{label}" for label in class_labels])
conformal_preds = pd.concat([conformal_preds.reset_index(drop=True), proba_df], axis=1)

# Merge with final dataset, including all columns that start with "proba"
results_df = final.merge(
    conformal_preds[
        [
            "eu_city_code",
            "predicted_class",
            "prediction_set",
            "prediction_set_size",
            "proba_low",
            "proba_moderate",
            "proba_high",
        ]
    ],
    on="eu_city_code",
    how="left",
)

results_df["prediction_set_size"].value_counts()
results_df[results_df["predicted"]]["eu_cycling_level"].value_counts(normalize=True) * 100

# Save results
results_df.to_csv(
    "models/clf_conformal_predictions.csv",
    index=False,
)
