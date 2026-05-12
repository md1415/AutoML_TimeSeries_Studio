import numpy as np
from .base_models import (
    XGBoostWrapper,
    ProphetWrapper,
    RandomForestWrapper,
    LSTMWrapper
)


class ModelSelector:
    def __init__(self, values=None):
        self.values = values
        self.models = {
            "short_term": XGBoostWrapper(),
            "seasonal": ProphetWrapper(),
            "baseline": RandomForestWrapper(),
            "lstm": None
        }

    def select_model(self, values: np.ndarray, dates: np.ndarray) -> str:
        n_points = len(values)
        self.values = values

        # Rule 1: Very small datasets -> XGBoost
        if n_points < 30:
            return "short_term"

        # Rule 2: Medium datasets (30-50) -> RandomForest
        if n_points < 50:
            return "baseline"

        # Rule 3: Large datasets (>=50) -> ALWAYS use LSTM first
        # This overrides Prophet for large datasets
        if n_points >= 50:
            return "lstm"

        # Rule 4: Seasonal detection (only for Prophet if LSTM fails)
        seasonal_strength = self._detect_seasonality(values)
        if seasonal_strength > 0.4 and n_points >= 60:
            return "seasonal"

        return "baseline"

    def _detect_seasonality(self, values: np.ndarray) -> float:
        n = len(values)
        if n < 14:
            return 0.0

        autocorr = np.correlate(values - np.mean(values),
                                values - np.mean(values),
                                mode='full'
                                )
        autocorr = autocorr[len(autocorr) // 2:]

        if len(autocorr) < 8:
            return 0.0

        lag_7 = autocorr[7] if len(autocorr) > 7 else 0
        lag_1 = autocorr[1] if len(autocorr) > 1 else 1

        if lag_1 == 0:
            return 0.0

        seasonal_ratio = abs(lag_7 / lag_1)
        return min(seasonal_ratio, 1.0)

    def get_model(self, model_key: str):
        if model_key == "lstm":
            if self.values is not None:
                lookback = max(5, min(30, int(len(self.values) * 0.15)))
                print(f"Initializing LSTM with lookback={lookback},"
                      f" data_points={len(self.values)}")
                return LSTMWrapper(lookback=lookback, epochs=50)
            else:
                return LSTMWrapper(lookback=10, epochs=50)

        return self.models.get(model_key, self.models["baseline"])
