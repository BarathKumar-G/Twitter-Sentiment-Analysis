"""
model.py - Logistic Regression model training for Twitter Sentiment Analysis.

Trains Logistic Regression on:
  - CountVectorizer features
  - TF-IDF features

Returns trained models and predictions for evaluation.
"""

from sklearn.linear_model import LogisticRegression


# ─── Model Configuration ─────────────────────────────────────────────────────

LR_PARAMS = dict(
    max_iter=1000,          # Ensure convergence
    solver="lbfgs",         # Memory-efficient for large feature spaces
    C=1.0,                  # Default regularisation strength
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
)


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, label: str = ""):
    """
    Fit a Logistic Regression model.

    Args:
        X_train: Sparse feature matrix for training.
        y_train: Training labels.
        label:   Descriptive string for logging (e.g., 'CountVectorizer').

    Returns:
        Fitted LogisticRegression model.
    """
    print(f"\n  Training Logistic Regression [{label}] ...")
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    print(f"  Training complete — classes: {list(model.classes_)}")
    return model


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict(model, X_test):
    """
    Generate hard predictions and probability scores.

    Returns:
        (y_pred, y_prob) — predicted class labels and probability of class 1.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # P(positive)
    return y_pred, y_prob
