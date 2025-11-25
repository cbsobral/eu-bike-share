import warnings
import json
import pandas as pd
from pathlib import Path
from src.model.funs.classifiers import tune_model
from src.report.funs._load_setup import load_setup

warnings.filterwarnings("ignore")


# -- CONFIG AND DATA --
target = "eu_cycling"
models_dir = Path("models/robustness")
data, _, _, info, _ = load_setup(task="clf")

# Experiment metadata
script_name = "MODELS"

# List of models to test (all except xgb)
models_to_test = ["gb", "lgb", "catboost", "logistic", "rf", "svm"]


# -- TUNE MODELS --
# Loop through each model
for model_name in models_to_test:
    # Tune model
    results = tune_model(
        data=data,
        model_name=model_name,
        dataset_name=info["dataset_name"],
    )

    # Create JSON log entry
    res = results[0]
    json_entry = {
        "name": f"---- {script_name} ----",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": res["target"],
        "model_name": res["model_name"],
        "dataset_name": res["dataset_name"],
        "accuracy": round(res["accuracy"], 3),
        "macro_f1": round(res["macro_f1"], 3),
    }

    # Append to shared JSONL file
    with open(models_dir / "robustness.jsonl", "a") as f:
        f.write(json.dumps(json_entry, indent=2) + "\n")
