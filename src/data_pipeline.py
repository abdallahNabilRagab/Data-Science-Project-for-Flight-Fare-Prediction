# src/data_pipeline.py
import os
import pandas as pd
import numpy as np

# ====================================================
# Default Data Path (relative to project root)
# ====================================================
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "Data_Train.xlsx"
)


class FlightDataPreprocessor:
    def __init__(self):
        print("🚀 Initializing FlightDataPreprocessor...")
        self.stop_mapping = {
            "non-stop": 0,
            "1 stop": 1,
            "2 stops": 2,
            "3 stops": 3,
            "4 stops": 4,
        }
        self.feature_columns = None
        print("✅ Preprocessor initialized successfully\n")

    # ====================================================
    # Load data
    # ====================================================
    def load_data(self, file_path: str | None = None) -> pd.DataFrame:
        path = file_path if file_path else DEFAULT_DATA_PATH
        print("📂 [LOAD DATA] Starting data loading...")
        print(f"📄 File path: {path}")

        df = pd.read_excel(path)

        print(f"✅ Data loaded successfully | Shape: {df.shape}\n")
        return df

    # ====================================================
    # Cleaning
    # ====================================================
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🧹 [CLEAN DATA] Cleaning data...")
        print(f"🔹 Shape before cleaning: {df.shape}")

        df = df.copy()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        print(f"✅ Cleaning done | Shape after cleaning: {df.shape}\n")
        return df

    # ====================================================
    # Date & Time Feature Engineering
    # ====================================================
    def engineer_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🕒 [DATETIME FEATURES] Engineering date & time features...")

        # Date_of_Journey
        df["Date_of_Journey"] = pd.to_datetime(
            df["Date_of_Journey"], format="%d/%m/%Y"
        )
        df["Journey_Day"] = df["Date_of_Journey"].dt.day
        df["Journey_Month"] = df["Date_of_Journey"].dt.month
        df.drop(columns=["Date_of_Journey"], inplace=True)

        # Dep_Time
        df["Dep_Time"] = pd.to_datetime(
            df["Dep_Time"], format="%H:%M", errors="coerce"
        )
        df["Dep_Hour"] = df["Dep_Time"].dt.hour
        df["Dep_Minute"] = df["Dep_Time"].dt.minute
        df.drop(columns=["Dep_Time"], inplace=True)

        # Arrival_Time
        time_only = df["Arrival_Time"].str.extract(r"(\d{1,2}:\d{2})")[0]
        parsed = pd.to_datetime(time_only, format="%H:%M", errors="coerce")
        df["Arrival_Hour"] = parsed.dt.hour
        df["Arrival_Minute"] = parsed.dt.minute
        df.drop(columns=["Arrival_Time"], inplace=True)

        print("✅ Datetime features engineered successfully\n")
        return df

    # ====================================================
    # Duration Feature Engineering
    # ====================================================
    def convert_duration(self, duration: str) -> int:
        hours, minutes = 0, 0
        if "h" in duration:
            hours = int(duration.split("h")[0])
        if "m" in duration:
            minutes = int(duration.split("m")[-2].split()[-1])
        return hours * 60 + minutes

    def engineer_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        print("⏱️ [DURATION] Converting duration to minutes...")

        df["Duration_Minutes"] = df["Duration"].apply(self.convert_duration)
        df.drop(columns=["Duration"], inplace=True)

        print("✅ Duration feature engineered successfully\n")
        return df

    # ====================================================
    # Encoding Categorical Features
    # ====================================================
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🏷️ [ENCODING] Encoding categorical features...")

        df["Total_Stops"] = df["Total_Stops"].map(self.stop_mapping)

        # Safe drop of 'Route' to avoid KeyError
        df.drop(columns=["Route"], inplace=True, errors="ignore")

        df = pd.get_dummies(
            df,
            columns=["Airline", "Source", "Destination", "Additional_Info"],
            drop_first=True,
        )

        print(f"✅ Encoding completed | Total features: {df.shape[1]}\n")
        return df

    # ====================================================
    # Fit / Transform (Training)
    # ====================================================
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔁 [FIT-TRANSFORM] Starting training preprocessing pipeline...")

        df = self.clean_data(df)
        df = self.engineer_datetime_features(df)
        df = self.engineer_duration(df)
        df = self.encode_features(df)

        self.feature_columns = df.drop(columns=["Price"]).columns
        print(f"💾 Feature columns saved for inference | Count: {len(self.feature_columns)}")
        print("✅ Training preprocessing completed successfully\n")

        # ====================================================
        # Save preprocessed data
        # ====================================================
        preprocessed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_path = os.path.join(preprocessed_dir, "train_preprocessed.parquet")
        df.to_parquet(preprocessed_path, index=False)
        print(f"💾 Preprocessed data saved to: {preprocessed_path}")

        return df

    # ====================================================
    # Transform (Inference)
    # ====================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 [TRANSFORM] Starting inference preprocessing...")

        df = self.clean_data(df)
        df = self.engineer_datetime_features(df)
        df = self.engineer_duration(df)
        df = self.encode_features(df)

        df = df.reindex(columns=self.feature_columns, fill_value=0)

        print("✅ Inference preprocessing completed successfully\n")
        return df
