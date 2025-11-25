import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.feature_selection import RFECV


# -- CONFIG --
# Setup paths
data_dir = Path("data/processed")
id_cols = ["eu_city_code", "eu_city", "eu_cycling"]
target_col = "eu_cycling"


# -- XGBOOST SELECTION --
# Load datasets
features_full = pd.read_csv(data_dir / "features_norm.csv")
train_data = features_full[features_full[target_col].notna()]

X = train_data.drop(columns=id_cols)
y = train_data[target_col]

# Cut (0-5%, 5-15%, above 15%)
bins = [float("-inf"), 0.05, 0.15, float("inf")]
labels = [0, 1, 2]
y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# XGBoost classifier
xgb_model = xgb.XGBClassifier()


# -- RFECV FEATURE SELECTION --
rfecv = RFECV(
    estimator=xgb_model,
    step=1,
    cv=5,
    scoring="accuracy",
    min_features_to_select=70,
)

rfecv.fit(X, y_class)

# Get selected features
selected_mask = rfecv.support_
rfecv_features = X.columns[selected_mask].tolist()

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {rfecv_features}")

# Save RFECV results
rfecv_df = pd.concat([features_full[id_cols], features_full[rfecv_features]], axis=1)
rfecv_df.to_csv(data_dir / "features_norm_rfecv.csv", index=False)
