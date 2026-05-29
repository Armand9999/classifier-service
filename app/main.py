from pathlib import Path
from typing import Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

APP_NAME = "routing-taxonomy-classifier-service"
APP_VERSION = "0.3.0"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "routing_taxonomy_6class_classifier.joblib"

app = FastAPI(title=APP_NAME, version=APP_VERSION)
model = None
ready = False


class ClassifyRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=300)
    text: str = Field(..., min_length=1, max_length=8000)


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    predicted_queue: str
    confidence: float
    probabilities: Dict[str, float]
    combined_text: str
    model_version: str
    requires_review: bool
    review_reason: str | None = None


def build_input_text(subject: str, text: str) -> str:
    s = " ".join((subject or "").strip().split())
    t = " ".join((text or "").strip().split())
    return f"{s} {t}".strip()


@app.on_event("startup")
def load_model() -> None:
    global model, ready
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    ready = True


@app.get("/health")
def health() -> dict:
    return {
        "service": APP_NAME,
        "status": "ok",
        "version": APP_VERSION,
        "taxonomy": "6-class-routing-taxonomy",
    }


@app.get("/ready")
def readiness() -> dict:
    if not ready or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {
        "service": APP_NAME,
        "status": "ready",
        "model_loaded": True,
        "model_path": MODEL_PATH.name,
        "version": APP_VERSION,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> ClassifyResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    combined_text = build_input_text(payload.subject, payload.text)
    if not combined_text:
        raise HTTPException(status_code=400, detail="Combined ticket text is empty")
    pred = model.predict([combined_text])[0]
    prob_values = model.predict_proba([combined_text])[0]
    classes = model.named_steps['clf'].classes_
    probabilities = {label: round(float(prob), 6) for label, prob in zip(classes, prob_values)}
    confidence = round(float(max(prob_values)), 6)
    requires_review = confidence < 0.75
    reason = None if not requires_review else ('manual-triage' if confidence < 0.50 else 'review-recommended')
    return ClassifyResponse(
        predicted_queue=str(pred),
        confidence=confidence,
        probabilities=probabilities,
        combined_text=combined_text,
        model_version=APP_VERSION,
        requires_review=requires_review,
        review_reason=reason,
    )
