import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.automl.model_selector import ModelSelector
from backend.automl.trainer import AutoMLTrainer

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
sessions = {}


class ForecastRequest(BaseModel):
    file_id: str
    horizon: int = 10
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    model_id: Optional[str] = None
    force_retrain: bool = False
    include_confidence: bool = True
    compare_models: bool = False


class ForecastResponse(BaseModel):
    status: str
    predictions: list
    confidence_intervals: Optional[Dict[str, list]] = None
    model_used: str
    training_info: dict
    horizon: int
    model_id: Optional[str] = None
    historical_data: Optional[Dict[str, list]] = None
    model_comparison: Optional[Dict[str, Any]] = None


class SavedModelInfo(BaseModel):
    model_id: str
    model_used: str
    data_points: int
    created_at: str
    seasonality_score: float


def calculate_confidence_intervals(predictions: np.ndarray, method: str = "bootstrap") -> Dict[str, list]:
    """Calculate simple confidence intervals based on prediction std"""
    if method == "simple":
        std_dev = np.std(predictions) * 0.1
        lower = predictions - 1.96 * std_dev
        upper = predictions + 1.96 * std_dev
    else:
        std_dev = np.std(predictions) * 0.15
        lower = predictions - 1.96 * std_dev
        upper = predictions + 1.96 * std_dev

    return {
        "lower": lower.tolist(),
        "upper": upper.tolist()
    }


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}.csv")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(file_path)

        if request.date_column and request.value_column:
            if request.date_column not in df.columns or request.value_column not in df.columns:
                raise HTTPException(status_code=400, detail="Specified columns not found")
            dates = pd.to_datetime(df[request.date_column]).values
            values = df[request.value_column].values.astype(float)
            date_strings = df[request.date_column].astype(str).tolist()
        else:
            if df.shape[1] < 2:
                raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")
            dates = pd.to_datetime(df.iloc[:, 0]).values
            values = df.iloc[:, 1].values.astype(float)
            date_strings = df.iloc[:, 0].astype(str).tolist()

        if len(values) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points")

        if request.horizon < 1 or request.horizon > 365:
            raise HTTPException(status_code=400, detail="Horizon must be between 1 and 365")

        historical_data = {
            "dates": date_strings[-min(30, len(date_strings)):],
            "values": values[-min(30, len(values)):].tolist()
        }

        session_key = f"{request.file_id}_{request.horizon}"
        model_comparison = None

        if request.compare_models:
            model_selector = ModelSelector()
            model_keys = ["short_term", "seasonal", "baseline"]
            comparison_predictions = {}

            for key in model_keys:
                temp_trainer = AutoMLTrainer(auto_save=False)
                temp_model = model_selector.get_model(key)
                scaled_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                temp_model.fit(dates, scaled_values)
                scaled_pred = temp_model.predict(request.horizon)
                original_pred = scaled_pred * (np.max(values) - np.min(values)) + np.min(values)
                comparison_predictions[key] = original_pred.tolist()

            model_comparison = {
                "xgb": comparison_predictions.get("short_term", []),
                "prophet": comparison_predictions.get("seasonal", []),
                "random_forest": comparison_predictions.get("baseline", [])
            }

        if not request.force_retrain and request.model_id:
            trainer = AutoMLTrainer(auto_save=False)
            trainer.load_model(request.model_id)
            training_info = {"status": "loaded_from_storage", "model_id": request.model_id}
            model_used = training_info.get("model_used", "loaded_model")
        elif session_key not in sessions or request.force_retrain:
            model_selector = ModelSelector()
            trainer = AutoMLTrainer(auto_save=True)
            training_info = trainer.train(dates, values, model_selector, save_model=True)
            sessions[session_key] = trainer
            model_used = training_info["model_used"]
        else:
            trainer = sessions[session_key]
            training_info = {"status": "loaded_from_cache"}
            model_used = trainer.model_name

        predictions = trainer.predict(request.horizon)

        response_data = {
            "status": "success",
            "predictions": predictions.tolist(),
            "model_used": model_used,
            "training_info": training_info,
            "horizon": request.horizon,
            "model_id": trainer.model_id if hasattr(trainer, 'model_id') else None,
            "historical_data": historical_data,
            "model_comparison": model_comparison
        }

        if request.include_confidence:
            response_data["confidence_intervals"] = calculate_confidence_intervals(predictions)

        return ForecastResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@router.get("/models", response_model=List[SavedModelInfo])
async def list_models():
    models = AutoMLTrainer.list_saved_models()
    return [
        SavedModelInfo(
            model_id=m['model_id'],
            model_used=m.get('model_used', 'unknown'),
            data_points=m.get('data_points', 0),
            created_at=m.get('created_at', ''),
            seasonality_score=m.get('seasonality_score', 0.0)
        )
        for m in models
    ]


@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    success = AutoMLTrainer.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "success", "message": f"Model {model_id} deleted"}