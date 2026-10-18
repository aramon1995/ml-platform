"""Online inference: the production model stays loaded in memory and predictions go to Postgres.

The model is loaded at startup and swapped by `POST /reload`, which the `select` flow calls after
every selection.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from mlplatform.config import get_settings
from mlplatform.inference import annotate, load_batch, load_production, predict, store

log = logging.getLogger(__name__)
state: dict[str, Any] = {}  # "current": (model, version), swapped as one value so requests never mix them


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)


class Prediction(BaseModel):
    predicted_value: str
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    model_version: int
    model_family: str
    feature_set: str
    required_columns: list[str]


class PredictResponse(ModelInfo):
    batch_id: str
    predictions: list[Prediction]


def load() -> ModelInfo:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    state["current"] = load_production()
    return model_info(*state["current"])


def model_info(model, version) -> ModelInfo:
    return ModelInfo(
        model_version=int(version.version),
        model_family=version.tags.get("model", "unknown"),
        feature_set=version.tags.get("feature_set", "unknown"),
        required_columns=list(model.feature_names_in_),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load()
    except Exception as e:  # the API starts anyway; /reload loads the model once one is promoted
        log.warning("No production model loaded: %s", e)
    yield


app = FastAPI(title="mlplatform", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Minimal page to try /predict by hand."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": model_info(*state["current"]) if state else None}


@app.post("/reload")
def reload() -> ModelInfo:
    try:
        return load()
    except Exception as e:
        raise HTTPException(503, f"Could not load the production model: {e}")


@app.post("/predict")
def predict_rows(request: PredictRequest) -> PredictResponse:
    if not state:
        raise HTTPException(503, "No production model loaded")
    model, version = state["current"]
    X, _ = load_batch(rows=request.rows)
    try:
        results = predict(model, X)
    except ValueError as e:
        raise HTTPException(422, str(e))
    batch_id = uuid.uuid4().hex
    store(annotate(results, version, batch_id, "api"))
    return PredictResponse(
        **model_info(model, version).model_dump(),
        batch_id=batch_id,
        predictions=results[["predicted_value", "probabilities"]].to_dict("records"),
    )
