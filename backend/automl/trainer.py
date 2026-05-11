import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from ..preprocessing.scaler import TimeSeriesScaler
from ..preprocessing.anomaly_detection import remove_anomalies
from .persistence import ModelPersistence


class AutoMLTrainer:
    def __init__(self, auto_save: bool = True):
        self.model = None
        self.scaler = TimeSeriesScaler()
        self.model_name = None
        self.is_trained = False
        self.auto_save = auto_save
        self.model_id = None
        self.training_info = {}

    def train(self, dates: np.ndarray, values: np.ndarray, model_selector, save_model: bool = True):
        cleaned_values = remove_anomalies(dates, values, method="linear")

        scaled_values = self.scaler.fit_transform(cleaned_values)

        model_key = model_selector.select_model(scaled_values, dates)
        self.model_name = model_key
        self.model = model_selector.get_model(model_key)

        self.model.fit(dates, scaled_values)
        self.is_trained = True

        self.training_info = {
            "model_used": self.model.name,
            "data_points": len(values),
            "anomalies_removed": int(np.sum(values != cleaned_values)),
            "seasonality_score": float(model_selector._detect_seasonality(scaled_values)),
            "model_key": model_key,
            "training_date": datetime.now().isoformat()
        }

        if save_model and self.auto_save:
            self.model_id = self._save_current_model()
            self.training_info["saved_model_id"] = self.model_id

        return self.training_info

    def _save_current_model(self) -> str:
        metadata = {
            **self.training_info,
            "scaler_mean": float(self.scaler.scaler.data_min_[0]) if hasattr(self.scaler, 'scaler') else None,
            "scaler_scale": float(self.scaler.scaler.scale_[0]) if hasattr(self.scaler, 'scaler') else None
        }

        return ModelPersistence.save_model(self.model, metadata)

    def load_model(self, model_id: str):
        model_data = ModelPersistence.load_model(model_id)
        metadata = ModelPersistence.load_metadata(model_id)

        self.model = model_data
        self.model_name = metadata.get("model_used")
        self.model_id = model_id
        self.is_trained = True
        self.training_info = metadata

        return metadata

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        scaled_predictions = self.model.predict(horizon)
        original_predictions = self.scaler.inverse_transform(scaled_predictions)
        return original_predictions

    @staticmethod
    def list_saved_models() -> list:
        return ModelPersistence.list_models()

    @staticmethod
    def delete_model(model_id: str) -> bool:
        return ModelPersistence.delete_model(model_id)