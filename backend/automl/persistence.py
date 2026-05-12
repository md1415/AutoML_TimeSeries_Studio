import os
import joblib
import json
from datetime import datetime
from typing import Dict, Any, Optional

MODEL_STORAGE_DIR = "saved_models"
os.makedirs(MODEL_STORAGE_DIR, exist_ok=True)


class ModelPersistence:
    @staticmethod
    def save_model(model,
                   metadata: Dict[str, Any],
                   model_id: Optional[str] = None) -> str:
        if model_id is None:
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        metadata_path = os.path.join(
            MODEL_STORAGE_DIR, f"{model_id}_metadata.json"
        )

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata['model_id'] = model_id
        metadata['created_at'] = datetime.now().isoformat()
        metadata['model_path'] = model_path

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_id

    @staticmethod
    def load_model(model_id: str):
        model_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_id} not found")

        return joblib.load(model_path)

    @staticmethod
    def load_metadata(model_id: str) -> Dict[str, Any]:
        metadata_path = os.path.join(
            MODEL_STORAGE_DIR, f"{model_id}_metadata.json"
        )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata for {model_id} not found")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def list_models() -> list:
        models = []
        for file in os.listdir(MODEL_STORAGE_DIR):
            if file.endswith('_metadata.json'):
                with open(os.path.join(MODEL_STORAGE_DIR, file), 'r') as f:
                    models.append(json.load(f))
        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    @staticmethod
    def delete_model(model_id: str) -> bool:
        model_path = os.path.join(MODEL_STORAGE_DIR, f"{model_id}.joblib")
        metadata_path = os.path.join(
            MODEL_STORAGE_DIR, f"{model_id}_metadata.json"
        )

        deleted = False
        if os.path.exists(model_path):
            os.remove(model_path)
            deleted = True
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        return deleted
