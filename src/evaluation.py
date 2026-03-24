"""
evaluation.py - Model evaluation and error analysis for Twitter Sentiment Analysis.

Provides:
- Accuracy, confusion matrix, and classification report
- Sample predictions printout
- Error analysis with misclassification examples
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


# ─── Label Mappings ──────────────────────────────────────────────────────────

LABEL_MAP = {0: "Negative", 1: "Positive"}


# ─── Core Evaluation ─────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_prob=None, model_name: str = "Model"):
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=["Negative", "Positive"])

    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC  : {auc:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':12s}  Pred Neg  Pred Pos")
    print(f"  {'Actual Neg':12s}  {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"  {'Actual Pos':12s}  {cm[1,0]:8d}  {cm[1,1]:8d}")

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Error Breakdown:")
    print(f"    FP: {fp}, FN: {fn}")

    if fp > fn:
        print("    → Bias towards Positive")
    elif fn > fp:
        print("    → Bias towards Negative")
    else:
        print("    → Balanced")

    print(f"\n  Classification Report:")
    for line in report.splitlines():
        print(f"    {line}")

    return acc, cm


# ─── Sample Predictions ───────────────────────────────────────────────────────

def show_sample_predictions(texts, y_true, y_pred, n: int = 8, model_name: str = ""):
    """
    Print n sample predictions showing: tweet → predicted vs. actual label.

    Args:
        texts:      Original tweet strings (for display).
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        n:          Number of samples to show.
        model_name: Label for the section header.
    """
    print(f"\n--- Sample Predictions [{model_name}] ---")
    indices = np.arange(len(texts))
    np.random.seed(42)
    chosen = np.random.choice(indices, size=min(n, len(texts)), replace=False)

    for i, idx in enumerate(chosen, 1):
        tweet   = str(texts[idx])[:90]         # Truncate long tweets
        actual  = LABEL_MAP.get(int(y_true[idx]), str(y_true[idx]))
        pred    = LABEL_MAP.get(int(y_pred[idx]), str(y_pred[idx]))
        correct = "PASS" if actual == pred else "FAIL"
        print(f"  [{i}] {correct} | Actual: {actual:<9} | Predicted: {pred:<9}")
        print(f"       Tweet: \"{tweet}\"")


# ─── Error Analysis ───────────────────────────────────────────────────────────

def error_analysis(texts, y_true, y_pred, n: int = 5, model_name: str = ""):
    """
    Show examples of misclassified tweets and explain likely causes.

    Analyses:
    - False Positives: predicted Positive, actual Negative
    - False Negatives: predicted Negative, actual Positive

    Args:
        texts:  Original tweet strings.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        n:      Number of examples per error type.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    texts  = np.array(texts)

    fp_mask = (y_pred == 1) & (y_true == 0)   # False Positives
    fn_mask = (y_pred == 0) & (y_true == 1)   # False Negatives

    fp_tweets = texts[fp_mask][:n]
    fn_tweets = texts[fn_mask][:n]

    print(f"\n{'='*60}")
    print(f"  Error Analysis [{model_name}]")
    print(f"  Total misclassified: {int(fp_mask.sum() + fn_mask.sum())} / {len(y_true)}")
    print(f"{'='*60}")

    print(f"\n  False Positives (predicted Positive, actually Negative):")
    print("  Likely causes: sarcasm, irony, negation ('not happy'), noise")
    for i, tweet in enumerate(fp_tweets, 1):
        print(f"    [{i}] \"{str(tweet)[:100]}\"")

    print(f"\n  False Negatives (predicted Negative, actually Positive):")
    print("  Likely causes: underrepresented positive phrases, rare slang, mixed sentiment")
    for i, tweet in enumerate(fn_tweets, 1):
        print(f"    [{i}] \"{str(tweet)[:100]}\"")

    # --- Dynamic Observations ---

    print(f"\n  Key Observations:")

    if fp_mask.sum() > fn_mask.sum():
        print("  - Model tends to over-predict Positive sentiment.")
    elif fn_mask.sum() > fp_mask.sum():
        print("  - Model tends to over-predict Negative sentiment.")
    else:
        print("  - Model errors are balanced between classes.")

    avg_length_fp = np.mean([len(str(t).split()) for t in texts[fp_mask]]) if fp_mask.any() else 0
    avg_length_fn = np.mean([len(str(t).split()) for t in texts[fn_mask]]) if fn_mask.any() else 0

    if avg_length_fp < avg_length_fn:
        print("  - Short tweets are more prone to misclassification.")
    elif avg_length_fn < avg_length_fp:
        print("  - Longer tweets show more misclassification.")

    if fp_mask.sum() + fn_mask.sum() == 0:
        print("  - No misclassifications detected (perfect performance on this set).")