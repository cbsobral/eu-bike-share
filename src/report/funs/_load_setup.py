import pandas as pd
from pathlib import Path
import joblib


def load_setup(
    task: str = "clf",
):
    """
    Load data, predictions, final predictions, best model info, and trained model for a given setup.

    Args:
        task (str, optional): "clf" for classification or "reg" for regression. Defaults to "clf".
        bin_option (str or None, optional): Binning option for classification. Required if task is "clf".

    Returns:
        tuple: (data, predictions, final, best_model, model, metrics)
            data (pd.DataFrame or None): Input dataset.
            predictions (pd.DataFrame): Model predictions.
            final (pd.DataFrame or None): Final predictions using all observations.
            best_model (pd.Series): Best model's metadata.
            model: Trained model object.
    """
    assert task in {"clf", "reg"}, "task must be 'clf' or 'reg'"
    task_name = "clf" if task == "clf" else "reg"

    models_dir = Path("models") if task == "clf" else Path("models/robustness")
    data_dir = Path("data/processed")

    # Load tuning results
    metrics_path = models_dir / f"{task_name}_metrics.csv"
    metrics = pd.read_csv(metrics_path)

    # Select best model
    if task == "clf":
        best_model_metrics = metrics.nlargest(1, ["accuracy", "macro_f1"]).iloc[0]
    else:  # task == "reg"
        best_model_metrics = metrics.nsmallest(1, "rmse").iloc[0]

    # Load predictions
    pred_path = models_dir / f"{task_name}_predictions.csv"
    predictions = pd.read_csv(pred_path)

    # Load final predictions if file exists
    final_path = models_dir / f"{task_name}_final_predictions.csv"
    conformal = pd.read_csv(final_path) if final_path.exists() else None

    # Load dataset
    dataset_name = best_model_metrics.get("dataset_name", None)
    data = pd.read_csv(data_dir / f"{dataset_name}.csv") if dataset_name else None

    # Load joblib model
    model_path = models_dir / f"{task_name}_model.joblib"
    model = joblib.load(model_path)

    return data, predictions, conformal, best_model_metrics, model
