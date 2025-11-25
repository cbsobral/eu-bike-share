import json
import pandas as pd
from src.model.funs.classifiers import tune_model
from src.report.funs._load_setup import load_setup
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


# -- CONFIG AND DATA --
target = "eu_cycling"
models_dir = Path("models/robustness")

# Experiment metadata
script_name = "BULGARIA"
data, _, _, info, _ = load_setup(task="clf")
data = data[~data["eu_city_code"].str.startswith("BG")]

# Tune model and get results
results = tune_model(
    data=data,
    model_name=info["model_name"],
    dataset_name=info["dataset_name"],
)


# -- JSON LOG --
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
