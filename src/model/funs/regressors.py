from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
import numpy as np
import pandas as pd
import joblib


# Define base models
def get_base_models():
    return {
        "xgb": XGBRegressor(random_state=42, objective="reg:pseudohubererror"),
    }


# Parameter space without loss parameter
def get_param_space(model_type):
    param_spaces = {
        "xgb": {
            "learning_rate": Real(0.001, 0.1, prior="log-uniform"),
            "max_depth": Integer(2, 6),
            "n_estimators": Integer(50, 200),
            "subsample": Real(0.6, 0.9),
            "colsample_bytree": Real(0.6, 0.9),
            "reg_alpha": Real(0.1, 10.0, prior="log-uniform"),
            "reg_lambda": Real(1.0, 20.0, prior="log-uniform"),
        },
    }
    return param_spaces.get(model_type, {})


# Cross-validation strategy
def get_cv_strategy(random_state=21):
    return KFold(n_splits=5, shuffle=True, random_state=random_state)


# Calculate model performance metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "error_std": np.std(np.abs(y_true - y_pred)),
        "n_neg_preds": np.sum(y_pred < 0),
    }
    return {k: round(v, 5) for k, v in metrics.items()}


# Prepare features and target
def prepare_features(df, target, transform_target=False):
    X = df.drop(columns=[target, "eu_city_code", "eu_city"])
    y_orig = df[target]

    if transform_target:
        # Log-transform target
        epsilon = 1e-6
        y = np.log(y_orig + epsilon)
    else:
        # Use original target
        y = y_orig.copy()

    return X, y, y_orig


# TUNE MODEL
def tune_model(
    data, model_type, dataset_name, target="eu_cycling", transform_target=False, iter=200
):
    # Prepare features and target
    data = data.dropna(subset=[target])
    X, y, y_orig = prepare_features(data, target, transform_target)

    # Skip models that cannot handle missing values
    if X.isna().any().any() and model_type in ["gb", "rf", "en", "svr"]:
        print(f"Skipping {model_type} - dataset contains NAs")
        return []

    # Initialize model and hyperparameter space
    model = get_base_models()[model_type]
    param_space = get_param_space(model_type)
    cv = get_cv_strategy()

    # Bayesian hyperparameter search (1 iteration for efficiency)
    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=21,
    )

    try:
        search.fit(X, y)
        best_model = search.best_estimator_

        # Predict using cross-validation
        y_pred_transformed = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)

        if transform_target:
            epsilon = 1e-6
            try:
                y_pred = np.exp(y_pred_transformed) - epsilon
            except (RuntimeWarning, OverflowError, FloatingPointError, ValueError):
                print(f"Error with exp transformation for {model_type}")
                return []
        else:
            # if negative return 0
            y_pred = np.where(y_pred_transformed < 0, 0, y_pred_transformed)

        # Compute performance metrics
        metrics = calculate_metrics(y_orig, y_pred)

        print(f"{model_type}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

        return [
            {
                "target": target,
                "model_name": model_type,
                "dataset_name": dataset_name,
                "transform": "log" if transform_target else "orig",
                "loss": "default",
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "neg_preds": (y_pred < 0).sum(),
                "min_pred": y_pred.min(),
                "params": best_model.get_params(),
                "y_pred": y_pred,
                "y_orig": y_orig,
                "eu_city_code": data["eu_city_code"],
                "model": best_model,
            }
        ]

    except Exception as e:
        print(f"Error training {model_type}: {str(e)}")
        return []


def save_results(results, models_dir):
    if not results:
        return

    # Create DataFrame and sort by RMSE and R2
    tuned_df = pd.DataFrame(results).sort_values(["rmse", "r2"], ascending=[True, False])

    # Just get the best model
    best_model = tuned_df.iloc[0]

    # Save predictions
    pd.DataFrame({
        "eu_city_code": best_model["eu_city_code"],
        "y_true": best_model["y_orig"],
        "y_pred": best_model["y_pred"],
    }).to_csv(models_dir / "reg_predictions.csv", index=False)

    # Save model
    joblib.dump(best_model["model"], models_dir / "reg_model.joblib")

    # Save all metrics (dropping the large objects)
    metrics_df = tuned_df.drop(
        columns=[
            "y_pred",
            "y_orig",
            "eu_city_code",
            "model",
            "params",
        ]
    )
    metrics_df.to_csv(models_dir / "reg_metrics.csv", index=False)

    return tuned_df
