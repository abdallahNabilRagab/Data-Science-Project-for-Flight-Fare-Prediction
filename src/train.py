# src/train.py
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from src import (
    FlightDataPreprocessor,
    NumericFeatureScaler,
    RFTopFeatureSelector,
    get_models,
    get_param_grids,
    evaluate_regression,
)

# =========================
# Numeric Features for scaling
# =========================
NUMERIC_FEATURES = [
    "Total_Stops",
    "Journey_Day",
    "Journey_Month",
    "Dep_Hour",
    "Dep_Minute",
    "Arrival_Hour",
    "Arrival_Minute",
    "Duration_Minutes"
]


def train_pipeline(model_dir: str, data_path: str | None = None):
    """
    Train all models inside a full pipeline (scaling + feature selection + model),
    select the best by RMSE, and save artifacts including feature columns.
    """
    print("\n🚀 ================= TRAINING PIPELINE STARTED =================")

    # =========================
    # Load + preprocess (RAW → TABULAR)
    # =========================
    print("\n📥 [STEP 1] Initializing data preprocessor...")
    preprocessor = FlightDataPreprocessor()

    print("📂 [STEP 2] Loading raw data...")
    df_raw = preprocessor.load_data(file_path=data_path)

    print("🧪 [STEP 3] Running preprocessing & feature engineering...")
    df = preprocessor.fit_transform(df_raw)
    print(f"✅ Preprocessing completed | Final dataset shape: {df.shape}")

    # =========================
    # Features & Target
    # =========================
    print("\n🎯 [STEP 4] Splitting features and target...")
    X = df.drop(columns=["Price"])
    y = df["Price"]
    print(f"🔹 Features shape: {X.shape}")
    print(f"🔹 Target shape: {y.shape}")

    # =========================
    # Train-Test Split
    # =========================
    print("\n✂️ [STEP 5] Performing train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Train size: {X_train.shape} | Test size: {X_test.shape}")

    # =========================
    # Models & Hyperparameters
    # =========================
    print("\n🧠 [STEP 6] Loading models and hyperparameter grids...")
    models = get_models()
    param_grids = get_param_grids()
    print(f"✅ Models loaded: {list(models.keys())}")

    best_rmse = float("inf")
    best_pipeline = None
    best_name = None

    # =========================
    # Train each model inside FULL pipeline
    # =========================
    print("\n🏋️ [STEP 7] Training models inside full pipeline...")
    for name, model in models.items():
        print(f"\n🔵 Training model: {name}")

        pipeline = Pipeline(steps=[
            ("scaler", NumericFeatureScaler(NUMERIC_FEATURES)),
            ("selector", RFTopFeatureSelector(n_features=25)),
            ("model", model)
        ])

        # Hyperparameter tuning if available
        if name in param_grids:
            print("⚙️  Running GridSearchCV...")
            grid = GridSearchCV(
                pipeline,
                param_grid={f"model__{k}": v for k, v in param_grids[name].items()},
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            trained_pipeline = grid.best_estimator_
            print("✅ GridSearch completed")
        else:
            print("⚙️  Training without hyperparameter tuning...")
            trained_pipeline = pipeline.fit(X_train, y_train)

        # =========================
        # Evaluation
        # =========================
        print("📊 Evaluating model on test set...")
        preds = trained_pipeline.predict(X_test)
        metrics = evaluate_regression(y_test, preds)

        print(
            f"📈 Results for {name} | "
            f"RMSE: {metrics['RMSE']:.2f} | "
            f"MAE: {metrics.get('MAE', 'N/A')} | "
            f"R2: {metrics.get('R2', 'N/A')}"
        )

        # Update best model
        if metrics["RMSE"] < best_rmse:
            print(f"🏆 New best model found: {name}")
            best_rmse = metrics["RMSE"]
            best_pipeline = trained_pipeline
            best_name = name

    # =========================
    # Save artifacts
    # =========================
    print("\n💾 [STEP 8] Saving best model, preprocessor, and feature columns...")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_pipeline.pkl")
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    feature_columns_path = os.path.join(model_dir, "feature_columns.pkl")

    joblib.dump(best_pipeline, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(list(X.columns), feature_columns_path)

    print(f"✅ Model pipeline saved to: {model_path}")
    print(f"✅ Preprocessor saved to: {preprocessor_path}")
    print(f"✅ Feature columns saved to: {feature_columns_path}")
    print(f"📝 Feature columns: {list(X.columns)}")

    # =========================
    # Final summary
    # =========================
    print("\n🎉 ================= TRAINING COMPLETED =================")
    print(f"🏆 Best Model: {best_name}")
    print(f"📉 Best RMSE: {best_rmse:.2f}\n")


# =========================
# Run training if file executed directly
# =========================
if __name__ == "__main__":
    print("▶️  Executing training script directly...")
    DEFAULT_MODEL_DIR = os.path.join(
        os.path.dirname(__file__), "..", "models", "saved_models"
    )
    train_pipeline(model_dir=DEFAULT_MODEL_DIR)
