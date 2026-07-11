from __future__ import annotations

from contextlib import asynccontextmanager
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    input: list[Any] = Field(..., description="1-D or 2-D numeric input")

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("'input' must not be empty")

        first_item = value[0]
        if isinstance(first_item, list):
            rows = value
            if not rows:
                raise ValueError("'input' must contain at least one row")
            row_len = len(rows[0])
            if row_len == 0:
                raise ValueError("rows in 'input' must not be empty")
            for row in rows:
                if not isinstance(row, list):
                    raise ValueError("'input' must be consistently 2-D")
                if len(row) != row_len:
                    raise ValueError("all rows in 'input' must have same length")
                for element in row:
                    if not isinstance(element, (int, float)):
                        raise ValueError("'input' values must be numeric")
            return value

        for element in value:
            if not isinstance(element, (int, float)):
                raise ValueError("'input' values must be numeric")
        return value


def _detect_model_type(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pkl", ".pickle", ".joblib"}:
        return "sklearn"
    raise RuntimeError(f"Unsupported model extension: {suffix}")


def _load_model(model_path: Path) -> tuple[Any, str]:
    model_type = _detect_model_type(model_path)

    if model_type == "onnx":
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return session, model_type

    if model_path.suffix.lower() == ".joblib":
        import joblib

        model = joblib.load(model_path)
        return model, model_type

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    return model, model_type


def _to_array(payload: PredictRequest) -> np.ndarray:
    input_data = payload.input
    first_item = input_data[0]
    if isinstance(first_item, list):
        return np.asarray(input_data, dtype=np.float32)
    return np.asarray(input_data, dtype=np.float32).reshape(-1, 1)


def _predict_onnx(session: Any, features: np.ndarray) -> dict[str, Any]:
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    outputs = session.run(None, {input_name: features})
    predictions = outputs[0]
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return {"predictions": predictions}


def _predict_sklearn(model: Any, features: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(features)
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()
    return {"predictions": predictions}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path_raw = os.environ.get("MODEL_PATH")
    if not model_path_raw:
        raise RuntimeError("MODEL_PATH env var is required")

    model_path = Path(model_path_raw)
    if not model_path.exists() or not model_path.is_file():
        raise RuntimeError(f"Model file not found: {model_path}")

    model, model_type = _load_model(model_path)
    app.state.model = model
    app.state.model_type = model_type
    app.state.model_path = str(model_path)
    yield


app = FastAPI(title="PRISM Model Container", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": getattr(app.state, "model_path", None),
        "model_type": getattr(app.state, "model_type", None),
    }


@app.get("/predict")
def predict_usage() -> dict[str, Any]:
    return {
        "detail": "Use POST /predict with JSON body.",
        "example_request": {
            "method": "POST",
            "path": "/predict",
            "json": {"input": [[1.0, 2.0]]},
        },
        "input_schema": PredictRequest.model_json_schema(),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    model = getattr(app.state, "model", None)
    model_type = getattr(app.state, "model_type", None)
    if model is None or model_type is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _to_array(payload)

    try:
        if model_type == "onnx":
            return _predict_onnx(model, features)
        if model_type == "sklearn":
            return _predict_sklearn(model, features)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc

    raise HTTPException(status_code=500, detail="Unknown model type")
