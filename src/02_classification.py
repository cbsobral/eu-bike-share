"""02_classification.py
tunes xgboost and six comparison classifiers and saves the xgboost model with all metrics
"""

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from utils import (
    OUTPUT_DIR,
    RANDOM_STATE,
    classification_metrics,
    get_cv_strategy,
    get_xgb_param_space,
    load_model_data,
)


# -- CONFIG --
# Models and search spaces
median = ("imputer", SimpleImputer(strategy="median"))
scaler = ("scaler", StandardScaler())
models = {
    "xgboost": (
        XGBClassifier(random_state=RANDOM_STATE, n_jobs=1),
        get_xgb_param_space(),
    ),
    "logistic_regression": (
        Pipeline(
            [
                median,
                scaler,
                (
                    "model",
                    LogisticRegression(random_state=RANDOM_STATE, max_iter=5000),
                ),
            ]
        ),
        {
            "model__C": Real(0.001, 100, prior="log-uniform"),
            "model__class_weight": Categorical(["balanced", None]),
        },
    ),
    "svm": (
        Pipeline([median, scaler, ("model", SVC(random_state=RANDOM_STATE))]),
        {
            "model__C": Real(0.1, 100, prior="log-uniform"),
            "model__gamma": Real(0.001, 1.0, prior="log-uniform"),
            "model__kernel": Categorical(["rbf", "linear"]),
            "model__class_weight": Categorical(["balanced", None]),
        },
    ),
    "random_forest": (
        Pipeline(
            [
                median,
                ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
            ]
        ),
        {
            "model__n_estimators": Integer(50, 500),
            "model__max_depth": Integer(2, 20),
            "model__min_samples_split": Integer(2, 20),
            "model__min_samples_leaf": Integer(1, 10),
            "model__max_features": Categorical(["sqrt", "log2", None]),
        },
    ),
    "gradient_boosting": (
        Pipeline(
            [median, ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]
        ),
        {
            "model__n_estimators": Integer(50, 500),
            "model__learning_rate": Real(0.001, 0.3, prior="log-uniform"),
            "model__max_depth": Integer(2, 10),
            "model__min_samples_leaf": Integer(1, 20),
            "model__subsample": Real(0.5, 1.0),
        },
    ),
    "lightgbm": (
        LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbosity=-1),
        {
            "n_estimators": Integer(50, 500),
            "learning_rate": Real(0.001, 0.3, prior="log-uniform"),
            "num_leaves": Integer(10, 100),
            "max_depth": Integer(3, 10),
            "min_child_samples": Integer(5, 30),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
            "reg_lambda": Real(1e-8, 10.0, prior="log-uniform"),
        },
    ),
    "catboost": (
        CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        ),
        {
            "iterations": Integer(50, 500),
            "learning_rate": Real(0.05, 0.3, prior="log-uniform"),
            "depth": Integer(4, 8),
            "l2_leaf_reg": Real(1.0, 5.0, prior="log-uniform"),
            "border_count": Integer(64, 128),
        },
    ),
}


# -- LOAD AND PREPARE --
_, labeled, features, y_class, y_encoded, encoder = load_model_data()
X = labeled[features]


# -- TUNE AND EVALUATE --
rows = []
for name, (base_model, search_space) in models.items():
    search = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_space,
        n_iter=100,
        cv=get_cv_strategy(),
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X, y_encoded)
    y_pred = encoder.inverse_transform(
        cross_val_predict(
            clone(base_model).set_params(**search.best_params_),
            X,
            y_encoded,
            cv=get_cv_strategy(),
            n_jobs=-1,
        )
    )
    rows.append({"model": name, **classification_metrics(y_class, y_pred)})

    # Main model
    if name == "xgboost":
        joblib.dump(search.best_estimator_, OUTPUT_DIR / "classification_model.joblib")


# -- SAVE --
comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
comparison.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
