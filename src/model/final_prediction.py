import pandas as pd
from pathlib import Path
from src.model.funs.classifiers import prepare_features, create_target_categories
from src.report.funs._load_setup import load_setup

# -- SETUP --
models_dir = Path("models")
data_dir = Path("data/processed")
target = "eu_cycling"


# -- LOAD AND PREPARE DATA --
data, _, _, _, saved_model = load_setup(task="clf")


# Load lookup_cities to get country
lookup_cities = pd.read_excel(
    "data/dictionaries/lookup_cities.xlsx", usecols=["eu_city_code", "eu_country"]
)

# -- PREDICT --
# Split data into known and unknown
train_mask = data[target].notna()
X_train, y_train = prepare_features(data[train_mask], target)
X_predict = data[~train_mask].drop(columns=[target, "eu_city_code", "eu_city"])

# Create categories for training data
y_cat, y_encoded, le = create_target_categories(y_train)


# Get predictions for missing values
y_pred_saved = le.inverse_transform(saved_model.predict(X_predict))

# Create final prediction dataset
final_df = data.copy()
final_df["eu_cycling_level"] = None
final_df.loc[train_mask, "eu_cycling_level"] = y_cat
final_df.loc[~train_mask, "eu_cycling_level"] = y_pred_saved
final_df["predicted"] = ~train_mask


# Merge with lookup_cities to get country information
final_df = final_df.merge(lookup_cities, on="eu_city_code", how="left")

# Reorder columns
id_cols = ["eu_city_code", "eu_city", "eu_country"]
target_cols = ["eu_cycling", "eu_cycling_level", "predicted"]
other_cols = [
    col for col in final_df.columns if col not in id_cols + target_cols + ["gei_cycling"]
]

final_df = final_df[id_cols + target_cols]


# -- SAVE --
final_df.to_csv(models_dir / "clf_final_predictions.csv", index=False)
