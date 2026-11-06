"""01_features.py
selects predictors by recursive feature elimination and saves the selected list
"""

from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

from utils import (
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET,
    create_target_classes,
    get_cv_strategy,
    load_data,
)


# -- LOAD AND PREPARE --
data, predictors, candidates = load_data()
labeled = data.loc[data[TARGET].notna()]
_, y_encoded, _ = create_target_classes(labeled[TARGET])


# -- SELECT FEATURES --
rfecv = RFECV(
    estimator=XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
    min_features_to_select=15,
    cv=get_cv_strategy(),
    scoring="accuracy",
    n_jobs=-1,
)
rfecv.fit(labeled[candidates], y_encoded)
selected = [feature for feature, keep in zip(candidates, rfecv.support_) if keep]


# -- SAVE --
OUTPUT_DIR.mkdir(exist_ok=True)
predictors.loc[predictors["variable"].isin(selected)].to_csv(
    OUTPUT_DIR / "selected_features.csv", index=False
)
