from pathlib import Path
from src.model.funs.regressors import (
    tune_model,
    save_results,
)
from src.report.funs._load_setup import load_setup


# -- CONFIG AND DATA--
target = "eu_cycling"
transform = False  # No log transform
models_dir = Path("models/robustness")

# Load required data
data, _, _, info, _ = load_setup(task="clf")


# -- TUNE MAIN MODEL --
results = tune_model(
    data=data,
    model_type=info["model_name"],
    dataset_name=info["dataset_name"],
    transform_target=transform,
)


# -- SAVE RESULTS --
if results:
    tuned_df, exp_name = save_results(results, target, models_dir)
