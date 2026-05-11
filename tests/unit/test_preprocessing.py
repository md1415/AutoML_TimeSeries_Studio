import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
import pandas as pd
from backend.preprocessing.anomaly_detection import detect_anomalies_iqr, remove_anomalies
from backend.preprocessing.scaler import TimeSeriesScaler


class TestAnomalyDetection:
    def test_detect_anomalies_no_anomalies(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        anomalies = detect_anomalies_iqr(data)
        assert not anomalies.any()

    def test_detect_anomalies_with_outliers(self):
        data = np.array([1, 2, 3, 100, 4, 5, 6, 7, 8, 9])
        anomalies = detect_anomalies_iqr(data)
        assert anomalies[3] == True

    def test_remove_anomalies_linear(self):
        data = np.array([1, 2, 3, 100, 4, 5, 6])
        cleaned = remove_anomalies(np.arange(len(data)), data, method="linear")
        assert cleaned[3] < 10


class TestTimeSeriesScaler:
    def test_fit_transform(self):
        scaler = TimeSeriesScaler()
        data = np.array([10, 20, 30, 40, 50])
        scaled = scaler.fit_transform(data)
        assert np.min(scaled) == 0
        assert np.max(scaled) == 1

    def test_inverse_transform(self):
        scaler = TimeSeriesScaler()
        original = np.array([10, 20, 30, 40, 50])
        scaled = scaler.fit_transform(original)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(original, recovered)