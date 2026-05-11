import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# Changed to absolute imports
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


class ForecastResponse(BaseModel):
    status: str
    predictions: list
    model_used: str
    training_info: dict
    horizon: int


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
        else:
            if df.shape[1] < 2:
                raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")
            dates = pd.to_datetime(df.iloc[:, 0]).values
            values = df.iloc[:, 1].values.astype(float)

        if len(values) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points")

        if request.horizon < 1 or request.horizon > 365:
            raise HTTPException(status_code=400, detail="Horizon must be between 1 and 365")

        session_key = f"{request.file_id}_{request.horizon}"

        if session_key not in sessions:
            model_selector = ModelSelector()
            trainer = AutoMLTrainer()
            training_info = trainer.train(dates, values, model_selector)
            sessions[session_key] = trainer
        else:
            trainer = sessions[session_key]
            training_info = {"status": "loaded_from_cache"}

        predictions = trainer.predict(request.horizon)

        return ForecastResponse(
            status="success",
            predictions=predictions.tolist(),
            model_used=trainer.model_name,
            training_info=training_info,
            horizon=request.horizon
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")