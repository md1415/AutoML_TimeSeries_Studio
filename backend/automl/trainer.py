import numpy as np
import pandas as pd
from ..preprocessing.scaler import TimeSeriesScaler
from ..preprocessing.anomaly_detection import remove_anomalies


class AutoMLTrainer:
    def __init__(self):
        self.model = None
        self.scaler = TimeSeriesScaler()
        self.model_name = None
        self.is_trained = False

    def train(self, dates: np.ndarray, values: np.ndarray, model_selector):
        cleaned_values = remove_anomalies(dates, values, method="linear")

        scaled_values = self.scaler.fit_transform(cleaned_values)

        model_key = model_selector.select_model(scaled_values, dates)
        self.model_name = model_key
        self.model = model_selector.get_model(model_key)

        self.model.fit(dates, scaled_values)
        self.is_trained = True

        return {
            "model_used": self.model.name,
            "data_points": len(values),
            "anomalies_removed": int(np.sum(values != cleaned_values)),
            "seasonality_score": float(model_selector._detect_seasonality(scaled_values))
        }

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        scaled_predictions = self.model.predict(horizon)
        original_predictions = self.scaler.inverse_transform(scaled_predictions)
        return original_predictions