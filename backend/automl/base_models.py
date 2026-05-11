import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet


class BaseModelWrapper:
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.fitted = False

    def fit(self, dates: np.ndarray, values: np.ndarray):
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError


class XGBoostWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.X_train = None

    def fit(self, dates: np.ndarray, values: np.ndarray):
        t = np.arange(len(values))
        X = t.reshape(-1, 1)
        self.X_train = X
        self.model.fit(X, values)
        self.fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        last_idx = len(self.X_train)
        future_t = np.arange(last_idx, last_idx + horizon).reshape(-1, 1)
        return self.model.predict(future_t)


class ProphetWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("Prophet")
        self.model = None
        self.last_date = None

    def fit(self, dates: np.ndarray, values: np.ndarray):
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': values
        })
        self.model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(df)
        self.last_date = df['ds'].max()
        self.fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        future_dates = pd.date_range(start=self.last_date, periods=horizon + 1, freq='D')[1:]
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values


class RandomForestWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.X_train = None

    def fit(self, dates: np.ndarray, values: np.ndarray):
        t = np.arange(len(values))
        X = t.reshape(-1, 1)
        self.X_train = X
        self.model.fit(X, values)
        self.fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        last_idx = len(self.X_train)
        future_t = np.arange(last_idx, last_idx + horizon).reshape(-1, 1)
        return self.model.predict(future_t)