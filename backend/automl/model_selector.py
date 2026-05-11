import numpy as np
from .base_models import XGBoostWrapper, ProphetWrapper, RandomForestWrapper


class ModelSelector:
    def __init__(self):
        self.models = {
            "short_term": XGBoostWrapper(),
            "seasonal": ProphetWrapper(),
            "baseline": RandomForestWrapper()
        }

    def select_model(self, values: np.ndarray, dates: np.ndarray) -> str:
        n_points = len(values)

        if n_points < 30:
            return "short_term"

        seasonal_strength = self._detect_seasonality(values)

        if seasonal_strength > 0.4 and n_points >= 60:
            return "seasonal"

        return "baseline"

    def _detect_seasonality(self, values: np.ndarray) -> float:
        n = len(values)
        if n < 14:
            return 0.0

        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
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
        return self.models.get(model_key, self.models["baseline"])