import os
import joblib

from src.preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "best_vectorizer.joblib")

LABEL_MAP = {0: "Negative", 1: "Positive"}

_model = None
_vectorizer = None


def _load():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(text: str) -> dict:
    _load()

    cleaned = clean_text(text)
    X = _vectorizer.transform([cleaned])
    pred = _model.predict(X)[0]
    prob = float(_model.predict_proba(X)[0][1])
    sentiment = LABEL_MAP.get(int(pred), str(pred))

    return {"sentiment": sentiment, "probability": prob}
