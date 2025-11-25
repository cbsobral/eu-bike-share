import json
import pandas as pd
from src.model.funs.classifiers import tune_model
from src.report.funs._load_setup import load_setup
from pathlib import Path


# -- CONFIG AND DATA --
target = "eu_cycling"
models_dir = Path("models/robustness")

# Experiment metadata
script_name = "THRESHOLDS"

data, _, _, info, _ = load_setup(task="clf")

# Baseline performance metrics
baseline_accuracy = 0.823
baseline_f1 = 0.788

# Threshold configurations
thresholds = [
    ("low_shift", 0.045, 0.145),
    ("high_shift", 0.055, 0.155),
]


# -- MODEL TUNING --
for label, mod, high in thresholds:
    results = tune_model(
        data=data,
        model_name=info["model_name"],
        dataset_name=f"{info['dataset_name']}_{label}",
        mod=mod,
        high=high,
    )

    # Extract results
    res = results[0]
    delta_accuracy = res["accuracy"] - baseline_accuracy
    delta_f1 = res["macro_f1"] - baseline_f1

    # Create JSON log entry
    json_entry = {
        "name": f"---- {script_name} ----",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": res["target"],
        "model_name": res["model_name"],
        "dataset_name": res["dataset_name"],
        "threshold_config": label,
        "mod_threshold": mod,
        "high_threshold": high,
        "accuracy": round(res["accuracy"], 3),
        "macro_f1": round(res["macro_f1"], 3),
        "delta_accuracy": round(delta_accuracy, 3),
        "delta_f1": round(delta_f1, 3),
    }

    # Append to shared JSONL file
    with open(models_dir / "robustness.jsonl", "a") as f:
        f.write(json.dumps(json_entry, indent=2) + "\n")
