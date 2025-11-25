import warnings
import pandas as pd
from pathlib import Path
from src.model.funs.classifiers import tune_model, save_results

warnings.filterwarnings("ignore")


# -- CONFIG AND DATA --
target = "eu_cycling"
models_dir = Path("models")

data = pd.read_csv("data/processed/features_norm_xgb.csv")

# -- TUNE MODEL --
results = tune_model(
    data=data,
    model_name="xgb",
    dataset_name="features_norm_xgb",
    target=target,
)

# -- SAVE RESULTS --
all_tuned_results = {target: [results]}
tuned_df = save_results(all_tuned_results[target], target, models_dir)
