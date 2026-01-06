# src/inference.py
import os
import joblib
import pandas as pd
from src import FlightDataPreprocessor

# ====================================================
# Default model directory
# ====================================================
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "saved_models"
)


def predict(input_df: pd.DataFrame, model_dir: str | None = None):
    """
    Generate predictions for new flight data using the saved pipeline.
    
    Parameters
    ----------
    input_df : pd.DataFrame
        Raw input data (same structure as original training data)
    model_dir : str | None
        Directory where model_pipeline.pkl and preprocessor.pkl are stored.
        If None, uses DEFAULT_MODEL_DIR.
    
    Returns
    -------
    np.ndarray
        Predicted flight prices.
    """
    model_dir = model_dir if model_dir else DEFAULT_MODEL_DIR
    print(f"📂 Loading model and preprocessor from: {model_dir}")

    # Load artifacts
    pipeline_path = os.path.join(model_dir, "model_pipeline.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")

    pipeline = joblib.load(pipeline_path)
    preprocessor = joblib.load(preprocessor_path)

    # Preprocess input and predict
    X = preprocessor.transform(input_df)
    predictions = pipeline.predict(X)

    return predictions


# =========================
# Example usage:
# from src.inference import predict
# df_new = pd.read_csv("new_flights.csv")
# preds = predict(df_new)
# =========================
