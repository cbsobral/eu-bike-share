from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pandas as pd
import joblib

import warnings

warnings.filterwarnings("ignore")


def prepare_features(df, target):
    X = df.drop(columns=[target, "eu_city_code", "eu_city"])
    y = df[target]

    return X, y


def get_cv_strategy(random_state=21):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def create_target_categories(y, mod=0.05, high=0.15):
    bins = [-float("inf"), mod, high, float("inf")]
    labels = ["low", "moderate", "high"]

    y_cat = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

    le = LabelEncoder()
    le.fit(labels)  # ensures all classes are known
    y_encoded = le.transform(y_cat)

    return y_cat, y_encoded, le


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    return {k: round(v, 3) for k, v in metrics.items()}


def tune_model(
    data,
    model_name,
    dataset_name,
    target="eu_cycling",
    n_iter=100,
    mod=0.05,
    high=0.15,
):
    results = []
    data = data.dropna(subset=[target])
    X, y = prepare_features(data, target)

    y_cat, y_encoded, le = create_target_categories(y, mod, high)

    # Model and search space
    base_model = get_base_models()[model_name]
    param_space = get_param_space(model_name)
    cv = get_cv_strategy()

    # Hyper-parameter search
    search = BayesSearchCV(
        estimator=base_model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        n_jobs=-1,
        random_state=42,
        refit="accuracy",
    )

    search.fit(X, y_encoded)
    best_accuracy = search.best_score_  # the refit metric
    best_f1 = search.cv_results_["mean_test_macro_f1"][search.best_index_]
    best_params = search.best_params_
    final_model = search.best_estimator_

    print(f"F1: {best_f1:.4f}, ACC: {best_accuracy:.4f}")

    # Cross-validated predictions with clone
    cv_model = clone(base_model).set_params(**best_params)
    y_pred_encoded = cross_val_predict(cv_model, X, y_encoded, cv=cv, n_jobs=-1)
    y_pred_cat = le.inverse_transform(y_pred_encoded)

    # Metrics
    metrics = calculate_metrics(y_cat, y_pred_cat)
    print(metrics["accuracy"], metrics["macro_f1"])

    # Assert metrics are similar (within 0.01 tolerance)
    assert abs(metrics["accuracy"] - best_accuracy) < 0.01, (
        f"Accuracy mismatch: {metrics['accuracy']:.4f} vs {best_accuracy:.4f}"
    )
    assert abs(metrics["macro_f1"] - best_f1) < 0.01, (
        f"F1 mismatch: {metrics['macro_f1']:.4f} vs {best_f1:.4f}"
    )

    # Prepare results dictionary
    results.append({
        "target": target,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "calc_accuracy": metrics["accuracy"],
        "calc_macro_f1": metrics["macro_f1"],
        "accuracy": best_accuracy,
        "macro_f1": best_f1,
        "y_true_num": y,
        "y_pred": y_pred_cat,
        "y_true": y_cat,
        "eu_city_code": data["eu_city_code"],
        "model": final_model,
    })
    return results


def save_results(results, target, version, models_dir):
    # Flatten the nested list of dictionaries
    flat_results = []
    for res_list in results:
        flat_results.extend(res_list)

    if not flat_results:
        return

    # Create DataFrame with all results
    tuned_df = pd.DataFrame(flat_results).sort_values(
        ["accuracy", "macro_f1"], ascending=False
    )

    # Save full metrics summary
    metrics_df = tuned_df.drop(
        columns=["y_pred", "y_true", "y_true_num", "eu_city_code", "model"]
    )
    metrics_df.to_csv(models_dir / f"{version}_clf_metrics.csv", index=False)

    # Get best model (single row as Series)
    best_model = tuned_df.loc[tuned_df["accuracy"].idxmax()]

    # Save predictions
    pred_df = pd.DataFrame({
        "eu_city_code": best_model["eu_city_code"],
        "y_true_num": best_model["y_true_num"],
        "y_true": best_model["y_true"],
        "y_pred": best_model["y_pred"],
    })
    pred_df.to_csv(
        models_dir / "clf_predictions.csv",
        index=False,
    )

    # Save model
    joblib.dump(best_model["model"], models_dir / "clf_model.joblib")

    return tuned_df


def get_base_models(random_state=42):
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=random_state)),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=random_state)),
        ]),
        "gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]),
        "xgb": XGBClassifier(random_state=random_state),
        "lgb": LGBMClassifier(random_state=random_state, verbose=-1),
        "catboost": CatBoostClassifier(
            verbose=False,
            random_state=random_state,
            train_dir=None,
            save_snapshot=False,  # Prevents creating snapshot files
            allow_writing_files=False,  # Completely disables file writing
            iterations=100,
        ),
    }


def get_param_space(model_type):
    param_spaces = {
        "logistic": {
            "model__C": Real(0.001, 100, prior="log-uniform"),
            "model__class_weight": Categorical(["balanced", None]),
        },
        "svm": {
            "model__C": Real(0.1, 100, prior="log-uniform"),
            "model__gamma": Real(0.001, 1.0, prior="log-uniform"),
            "model__kernel": Categorical(["rbf", "linear"]),
            "model__class_weight": Categorical(["balanced", None]),
        },
        "rf": {
            "model__n_estimators": Integer(50, 500),
            "model__max_depth": Integer(2, 20),
            "model__min_samples_split": Integer(2, 20),
            "model__min_samples_leaf": Integer(1, 10),
            "model__max_features": Categorical(["sqrt", "log2", None]),
        },
        "gb": {
            "model__n_estimators": Integer(50, 500),
            "model__learning_rate": Real(0.001, 0.3, prior="log-uniform"),
            "model__max_depth": Integer(2, 10),
            "model__min_samples_leaf": Integer(1, 20),
            "model__subsample": Real(0.5, 1.0),
        },
        "xgb": {
            "n_estimators": Integer(50, 500),
            "learning_rate": Real(0.001, 0.3, prior="log-uniform"),
            "max_depth": Integer(2, 10),
            "min_child_weight": Integer(1, 7),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
            "reg_lambda": Real(1e-8, 10.0, prior="log-uniform"),
        },
        "lgb": {
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
        "catboost": {
            "learning_rate": Real(0.05, 0.3, prior="log-uniform"),
            "depth": Integer(4, 8),
            "l2_leaf_reg": Real(1.0, 5.0, prior="log-uniform"),
            "border_count": Integer(64, 128),
        },
    }
    return param_spaces.get(model_type, {})
