import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from src.model.funs.classifiers import (
    prepare_features,
    create_target_categories,
    tune_model,
)
from src.report.funs._load_setup import load_setup


# -- CONFIG AND DATA --
target = "eu_cycling"
models_dir = Path("models/robustness")

# Experiment metadata
script_name = "COUNTRIES"

data, _, _, info, _ = load_setup(task="clf")
data = data.dropna(subset=[target])
geo = pd.read_excel("data/dictionaries/lookup_cities.xlsx")


# -- FUNCTION TO RUN EXPERIMENT WITH RETRAINING --
def run_experiment_with_retraining(data, geo, countries, model_name="xgb"):
    for country in countries:
        # Country-based split
        test_idx = geo.loc[geo["eu_country_code"] == country, "eu_city_code"]
        mask_test = data["eu_city_code"].isin(test_idx)
        train_data = data[~mask_test].copy()
        test_data = data[mask_test].copy()

        # Use tune_model function on resampled training data
        tune_results = tune_model(
            data=train_data,
            model_name=info["model_name"],
            dataset_name=info["dataset_name"],
        )

        if not tune_results:
            continue

        # Get the trained model
        trained_model = tune_results[0]["model"]

        # Prepare test features and evaluate
        X_test, y_test_cont = prepare_features(test_data, target=target)
        y_test_cat, *_ = create_target_categories(y_test_cont)

        # Make predictions
        y_pred_encoded = trained_model.predict(X_test)

        # Get label encoder from training data to inverse transform
        _, _, le = create_target_categories(
            prepare_features(train_data, target=target)[1]
        )
        y_pred_cat = le.inverse_transform(y_pred_encoded)

        # Calculate metrics
        acc = accuracy_score(y_test_cat, y_pred_cat)
        f1_macro = f1_score(y_test_cat, y_pred_cat, average="macro", zero_division=0)
        f1_weighted = f1_score(
            y_test_cat, y_pred_cat, average="weighted", zero_division=0
        )

        # Create JSON log entry
        json_entry = {
            "name": f"---- {script_name} ----",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target": target,
            "model_name": model_name,
            "dataset_name": info["dataset_name"],
            "test_country": country,
            "n_test_cities": len(test_data),
            "n_train_cities": len(train_data),
            "accuracy": round(acc, 3),
            "macro_f1": round(f1_macro, 3),
            "weighted_f1": round(
                f1_weighted,
                3,
            ),
        }
        # Append to shared JSONL file
        with open(models_dir / "robustness.jsonl", "a") as f:
            f.write(json.dumps(json_entry, indent=2) + "\n")


# -- RUN EXPERIMENTS --
countries = ["NL", "PT", "BE"]
run_experiment_with_retraining(data, geo, countries, model_name="xgb")
