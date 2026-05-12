import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler


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
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
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
        import pandas as pd
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': values
        })
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.model.fit(df)
        self.last_date = df['ds'].max()
        self.fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted")
        import pandas as pd
        future_dates = pd.date_range(
            start=self.last_date,
            periods=horizon + 1,
            freq='D')[1:]
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


class LSTMWrapper(BaseModelWrapper):
    def __init__(self, lookback: int = 10, epochs: int = 30):
        super().__init__("LSTM")
        self.lookback = lookback
        self.epochs = epochs
        self.model = None
        self.scaler = MinMaxScaler()
        self.last_sequence = None

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def fit(self, dates: np.ndarray, values: np.ndarray):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            raise ImportError(
                "TensorFlow not installed."
                " Install with: pip install tensorflow")

        # Normalize data
        scaled_values = (
            self.scaler.fit_transform(values.reshape(-1, 1)).flatten()
        )

        # Create sequences
        X, y = self._create_sequences(scaled_values)

        if len(X) == 0:
            raise ValueError(
                f"Not enough data for LSTM."
                f" Need at least {self.lookback + 1} points")

        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build LSTM model
        self.model = Sequential([
            LSTM(50, activation='relu',
                 return_sequences=True,
                 input_shape=(self.lookback, 1)
                 ),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')

        # Early stopping
        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )

        # Train
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=8,
            verbose=0,
            callbacks=[early_stop]
        )

        # Store last sequence for prediction
        self.last_sequence = scaled_values[-self.lookback:]
        self.fitted = True

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained first")

        if self.last_sequence is None:
            raise ValueError("No sequence available for prediction")

        predictions = []
        current_sequence = self.last_sequence.copy()

        for _ in range(horizon):
            # Reshape for prediction
            input_seq = current_sequence.reshape((1, self.lookback, 1))

            # Predict next value (scaled)
            next_scaled = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_scaled)

            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_scaled)

        # Inverse transform to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        original_predictions = (
            self.scaler.inverse_transform(predictions).flatten()
        )

        return original_predictions
