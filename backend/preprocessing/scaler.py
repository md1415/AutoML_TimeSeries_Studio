import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesScaler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False

    def fit(self, data: np.ndarray):
        self.scaler.fit(data.reshape(-1, 1))
        self.fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        return self.scaler.transform(data.reshape(-1, 1)).flatten()

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)