"""utils.py
shared data loading, cycling classes, folds and the xgboost search space for the pipeline scripts
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real


OUTPUT_DIR = Path("output")
TARGET = "cycling_mode_share"
LABELS = ["low", "moderate", "high"]
RANDOM_STATE = 21


def load_data():
    """Load the model input and the predictors eligible for selection"""
    data = pd.read_csv("data/features.csv")
    predictors = pd.read_csv("data/lookup_variables.csv")
    candidates = predictors["variable"].tolist()
    return data, predictors, candidates


def load_model_data():
    """Load labeled cities, the selected predictors and their classes"""
    data, _, _ = load_data()
    labeled = data.loc[data[TARGET].notna()].copy()
    features = pd.read_csv(OUTPUT_DIR / "selected_features.csv")["variable"].tolist()
    y_class, y_encoded, encoder = create_target_classes(labeled[TARGET])
    return data, labeled, features, y_class, y_encoded, encoder


def create_target_classes(y, moderate=0.05, high=0.15):
    """Cut cycling mode share into low, moderate and high classes"""
    classes = pd.cut(
        y,
        bins=[float("-inf"), moderate, high, float("inf")],
        labels=LABELS,
        include_lowest=True,
    )
    encoder = LabelEncoder().fit(LABELS)
    return classes, encoder.transform(classes), encoder


def get_cv_strategy():
    """Return the five-fold stratified split used in every model"""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def get_xgb_param_space():
    """Return the xgboost search space"""
    return {
        "n_estimators": Integer(50, 500),
        "learning_rate": Real(0.001, 0.3, prior="log-uniform"),
        "max_depth": Integer(2, 10),
        "min_child_weight": Integer(1, 7),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_lambda": Real(1e-8, 10.0, prior="log-uniform"),
    }


def classification_metrics(y_true, y_pred):
    """Return accuracy, macro f1 and the f1 of each class"""
    class_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        **{f"f1_{label}": value for label, value in zip(LABELS, class_f1)},
    }
