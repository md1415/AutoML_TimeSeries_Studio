import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.automl.model_selector import ModelSelector
from backend.automl.trainer import AutoMLTrainer
from .export import set_prediction_cache

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


def calculate_confidence_intervals(
        predictions: np.ndarray,
        method: str = "bootstrap") -> Dict[str, list]:
    if method == "simple":
        std_dev = np.std(predictions) * 0.1
        lower = predictions - 1.96 * std_dev
        upper = predictions + 1.96 * std_dev
    else:
        std_dev = np.std(predictions) * 0.15
        lower = predictions - 1.96 * std_dev
        upper = predictions + 1.96 * std_dev
    return {"lower": lower.tolist(), "upper": upper.tolist()}


# ========== helper functions to reduce complexity ==========

def _validate_data(df, request):
    """Validate CSV and extract data"""
    if request.date_column and request.value_column:
        # flake8: noqa: E503
        if (request.date_column not in df.columns
                or request.value_column not in df.columns):
            raise HTTPException(
                status_code=400,
                detail="Specified columns not found"
            )
        dates = pd.to_datetime(df[request.date_column]).values
        values = df[request.value_column].values.astype(float)
        date_strings = df[request.date_column].astype(str).tolist()
    else:
        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail="CSV must have at least 2 columns"
            )
        dates = pd.to_datetime(df.iloc[:, 0]).values
        values = df.iloc[:, 1].values.astype(float)
        date_strings = df.iloc[:, 0].astype(str).tolist()

    if len(values) < 10:
        raise HTTPException(
            status_code=400,
            detail="Need at least 10 data points"
        )

    if request.horizon < 1 or request.horizon > 365:
        raise HTTPException(
            status_code=400,
            detail="Horizon must be between 1 and 365"
        )

    return dates, values, date_strings


def _run_all_models_comparison(values, dates, horizon):
    """Run comparison for all 4 models: XGBoost, Prophet, RandomForest, LSTM"""
    model_selector = ModelSelector()

    model_configs = [
        ("short_term", "XGBoost"),
        ("seasonal", "Prophet"),
        ("baseline", "Random Forest"),
        ("lstm", "LSTM")
    ]

    comparison_predictions = {}
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    range_val = max_val - min_val if max_val - min_val > 0 else 1.0

    for model_key, model_name in model_configs:
        try:
            temp_model = model_selector.get_model(model_key)
            scaled_values = (values - min_val) / range_val
            temp_model.fit(dates, scaled_values)
            scaled_pred = temp_model.predict(horizon)
            original_pred = scaled_pred * range_val + min_val
            comparison_predictions[model_name] = original_pred.tolist()
        except Exception as e:
            print(f"Model {model_name} failed: {str(e)}")
            comparison_predictions[model_name] = [None] * horizon

    return comparison_predictions


def _get_or_train_model(session_key, request, dates, values):
    """Get existing model from cache or train a new one"""
    if not request.force_retrain and request.model_id:
        trainer = AutoMLTrainer(auto_save=False)
        trainer.load_model(request.model_id)
        training_info = {
            "status": "loaded_from_storage",
            "model_id": request.model_id
        }
        model_used = training_info.get("model_used", "loaded_model")
    elif session_key not in sessions or request.force_retrain:
        model_selector = ModelSelector()
        trainer = AutoMLTrainer(auto_save=True)
        training_info = trainer.train(
            dates,
            values,
            model_selector,
            save_model=True
        )
        sessions[session_key] = trainer
        model_used = training_info["model_used"]
    else:
        trainer = sessions[session_key]
        training_info = {"status": "loaded_from_cache"}
        model_used = trainer.model_name

    return trainer, training_info, model_used


# ========== main endpoint ==========

@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}.csv")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(file_path)
        dates, values, date_strings = _validate_data(df, request)

        historical_data = {
            "dates": date_strings[-min(30, len(date_strings)):],
            "values": values[-min(30, len(values)):].tolist()
        }

        model_comparison = None
        if request.compare_models:
            model_comparison = _run_all_models_comparison(
                values,
                dates,
                request.horizon
            )

        session_key = f"{request.file_id}_{request.horizon}"
        trainer, training_info, model_used = _get_or_train_model(
            session_key,
            request,
            dates,
            values
        )

        predictions = trainer.predict(request.horizon)
        set_prediction_cache(request.file_id, predictions.tolist(), model_used)

        response_data = {
            "status": "success",
            "predictions": predictions.tolist(),
            "model_used": model_used,
            "training_info": training_info,
            "horizon": request.horizon,
            "model_id": (
                trainer.model_id if hasattr(trainer, 'model_id')
                else None
            ),
            "historical_data": historical_data,
            "model_comparison": model_comparison
        }

        if request.include_confidence:
            response_data["confidence_intervals"] = (
                calculate_confidence_intervals(predictions)
            )

        return ForecastResponse(**response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {str(e)}"
        )


# ========== utility endpoints ==========

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
