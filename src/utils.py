# src/utils.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =========================
# Numeric Feature Scaler
# =========================
class NumericFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features):
        self.numeric_features = numeric_features
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.numeric_features])
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out[self.numeric_features] = self.scaler.transform(
            X_out[self.numeric_features]
        )
        return X_out


# =========================
# Feature Selector (RF-based)
# =========================
class RFTopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=25, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_ = None

    def fit(self, X, y):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y)

        importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        self.selected_features_ = importances.head(self.n_features).index.tolist()
        return self

    def transform(self, X):
        return X[self.selected_features_]
