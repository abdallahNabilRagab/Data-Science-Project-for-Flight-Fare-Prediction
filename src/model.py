# src/model.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

def get_param_grids():
    return {
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [8, 10],
            "min_samples_split": [2, 5]
        },
        "GradientBoosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }
    }
