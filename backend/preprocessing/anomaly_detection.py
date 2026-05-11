import numpy as np
import pandas as pd


def detect_anomalies_iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    anomalies = (data < lower_bound) | (data > upper_bound)
    return anomalies


def remove_anomalies(data: np.ndarray, values: np.ndarray, method: str = "linear") -> np.ndarray:
    anomalies = detect_anomalies_iqr(values)
    if not anomalies.any():
        return values

    cleaned = values.copy()
    anomaly_indices = np.where(anomalies)[0]

    if method == "linear":
        for idx in anomaly_indices:
            if idx == 0 or idx == len(values) - 1:
                cleaned[idx] = np.median(values[~anomalies]) if len(values[~anomalies]) > 0 else values[idx]
            else:
                cleaned[idx] = (values[idx - 1] + values[idx + 1]) / 2

    return cleaned