"""05_regression.py
tunes the xgboost regression of cycling mode share and saves the model with its metrics
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from utils import (
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET,
    get_cv_strategy,
    get_xgb_param_space,
    load_model_data,
)


# -- LOAD AND PREPARE --
_, labeled, features, _, y_encoded, _ = load_model_data()
X = labeled[features]
y = labeled[TARGET]
cv_splits = list(get_cv_strategy().split(X, y_encoded))  # same folds as the classifier


# -- TUNE AND EVALUATE --
base_model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1)
search = BayesSearchCV(
    estimator=base_model,
    search_spaces=get_xgb_param_space(),
    n_iter=50,
    cv=cv_splits,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
search.fit(X, y)
tuned_model = clone(base_model).set_params(**search.best_params_)
y_pred = np.clip(cross_val_predict(tuned_model, X, y, cv=cv_splits, n_jobs=-1), 0, None)
metrics = pd.DataFrame(
    [{"rmse": mean_squared_error(y, y_pred) ** 0.5, "r2": r2_score(y, y_pred)}]
)


# -- SAVE --
joblib.dump(search.best_estimator_, OUTPUT_DIR / "regression_model.joblib")
metrics.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
