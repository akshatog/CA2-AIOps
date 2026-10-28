"""Serve the fitted Colab pipeline. No training occurs in this application."""
import json
import math
import platform
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


class WineMeasurements(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(gt=0)
    ph: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0, le=100)


class Prediction(BaseModel):
    predicted_quality: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("model.joblib", "metadata.json"):
        if not (ARTIFACT_DIR / name).is_file():
            raise RuntimeError(
                f"Missing artifacts/{name}. Extract your Colab artifacts ZIP into artifacts/."
            )

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8-sig"))
    current_python = ".".join(platform.python_version().split(".")[:2])
    if current_python != metadata["python_major_minor"]:
        raise RuntimeError(
            f"Use Python {metadata['python_major_minor']} to match Colab; "
            f"this environment uses {current_python}."
        )
    mismatches = []
    for package, expected in metadata["package_versions"].items():
        installed = version(package)
        if installed != expected:
            mismatches.append(f"{package}: installed {installed}, expected {expected}")
    if mismatches:
        raise RuntimeError(
            "Install the exported artifacts/requirements-model.txt in your virtual environment. "
            + "; ".join(mismatches)
        )

    if metadata["features"] != list(WineMeasurements.model_fields):
        raise RuntimeError("The saved feature list does not match this wine API schema.")

    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    if list(model.feature_names_in_) != metadata["features"]:
        raise RuntimeError("The model and metadata have different input features.")

    app.state.model = model
    app.state.metadata = metadata
    yield
    del app.state.model


app = FastAPI(
    title="Red Wine Quality Prediction API",
    description="Estimate a red wine's quality score from 11 laboratory measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info(request: Request):
    metadata = request.app.state.metadata
    return {
        "task": metadata["task"],
        "selected_model": metadata["selected_model"],
        "features": metadata["features"],
    }


@app.post("/predict", response_model=Prediction)
def predict(measurements: WineMeasurements, request: Request):
    features = request.app.state.metadata["features"]
    row = pd.DataFrame([measurements.model_dump()], columns=features)
    try:
        value = float(request.app.state.model.predict(row)[0])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="The model could not process these measurements.") from exc
    if not math.isfinite(value):
        raise HTTPException(status_code=500, detail="The model returned an invalid prediction.")
    return Prediction(predicted_quality=value)
