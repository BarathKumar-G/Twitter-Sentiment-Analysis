import os
import joblib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "best_vectorizer.joblib")

LABEL_MAP = {0: "Negative", 1: "Positive"}

model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    yield


app = FastAPI(title="Twitter Sentiment Analysis API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    probability: float


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text must not be empty.")

    cleaned = clean_text(request.text)

    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="Input text is empty after cleaning.")

    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    prob = float(model.predict_proba(X)[0][1])
    sentiment = LABEL_MAP.get(int(pred), str(pred))

    return PredictResponse(sentiment=sentiment, probability=prob)
