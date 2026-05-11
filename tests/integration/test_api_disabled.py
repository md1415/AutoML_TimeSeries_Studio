import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


class TestAPI:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_upload_invalid_file(self):
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"content", "text/plain")}
        )
        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_forecast_without_upload(self):
        response = client.post(
            "/api/forecast",
            json={"file_id": "nonexistent", "horizon": 10}
        )
        assert response.status_code == 404

    def test_list_models_endpoint(self):
        response = client.get("/api/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)