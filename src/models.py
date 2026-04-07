from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

LR_PARAMS = dict(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    random_state=42,
)

NB_PARAMS = dict(
    alpha=1.0,
)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(**LR_PARAMS),
        "Naive Bayes": MultinomialNB(**NB_PARAMS),
    }


def train_model(model, X_train, y_train, label: str = ""):
    model_name = type(model).__name__
    print(f"\n  Training {model_name} [{label}] ...")
    model.fit(X_train, y_train)
    print(f"  Training complete - classes: {list(model.classes_)}")
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob
