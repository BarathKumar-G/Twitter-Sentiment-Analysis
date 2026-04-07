import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import nltk

nltk.download("wordnet", quiet=True)

import pandas as pd
from src.preprocessing import load_data, preprocess, clean_text
from src.features       import build_count_features, build_tfidf_features
from src.models         import get_models, train_model, predict
from src.evaluation     import evaluate_model, show_sample_predictions, error_analysis, compare_models


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "train_data.csv")
TEST_PATH   = os.path.join(BASE_DIR, "test_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_SAMPLE = 200_000


def separator(title: str = ""):
    width = 60
    if title:
        line = f"{'-'*4} {title} {'-'*(width - len(title) - 6)}"
    else:
        line = "=" * width
    print(f"\n{line}")


def step1_load_data():
    separator("STEP 1: DATA LOADING")

    print("\n[Train]")
    train_df = load_data(TRAIN_PATH, sample_size=TRAIN_SAMPLE)

    print("\n[Test]")
    test_df = load_data(TEST_PATH, sample_size=None)

    if len(test_df) < 3000:
        print("\n[Supplementing Test Data]")
        extra_test_df = train_df.sample(n=3000, random_state=42)
        train_df = train_df.drop(extra_test_df.index)
        test_df = pd.concat([test_df, extra_test_df]).reset_index(drop=True)
        print(f"  Supplemented test set with 3000 samples from train data. New test size: {len(test_df)}")

    return train_df, test_df


def step2_preprocess(train_df, test_df):
    separator("STEP 2: PREPROCESSING")

    print("\n  Cleaning train data ...")
    train_df = preprocess(train_df)

    print("  Cleaning test data ...")
    test_df  = preprocess(test_df)

    return train_df, test_df


def step3_eda(train_df):
    separator("STEP 3: EXPLORATORY DATA ANALYSIS")

    print("\n  Class Distribution (train):")
    dist  = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    for label, count in dist.items():
        name = "Negative" if label == 0 else "Positive"
        pct  = count / total * 100
        print(f"    {name} ({label}): {count:,}  ({pct:.1f}%)")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    plt.bar(["Negative", "Positive"], [dist.get(0, 0), dist.get(1, 0)], color=["red", "green"])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "class_distribution.png"))
    plt.close()

    train_df["tweet_len"] = train_df["clean_text"].str.split().str.len()
    print("\n  Tweet Length Stats:")
    print(f"    min   : {np.min(train_df['tweet_len']):.1f}")
    print(f"    max   : {np.max(train_df['tweet_len']):.1f}")
    print(f"    mean  : {np.mean(train_df['tweet_len']):.1f}")
    print(f"    median: {np.median(train_df['tweet_len']):.1f}")

    plt.figure(figsize=(8, 5))
    plt.hist(train_df["tweet_len"].dropna(), bins=50, color="blue", alpha=0.7)
    plt.title("Tweet Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "tweet_length_distribution.png"))
    plt.close()


def step4_features(train_df, test_df):
    separator("STEP 4: FEATURE EXTRACTION")

    train_texts = train_df["clean_text"].values
    test_texts  = test_df["clean_text"].values

    print("\n  [CountVectorizer]")
    X_train_cv, X_test_cv, cv_vec = build_count_features(train_texts, test_texts)

    print("\n  [TF-IDF]")
    X_train_tfidf, X_test_tfidf, tfidf_vec = build_tfidf_features(train_texts, test_texts)

    print("\n  Top 20 Words (after vectorization):")
    sums        = np.array(X_train_cv.sum(axis=0)).flatten()
    words       = cv_vec.get_feature_names_out()
    top_indices = sums.argsort()[::-1][:20]
    top_words   = [words[i] for i in top_indices]
    top_freqs   = [sums[i] for i in top_indices]

    for word, freq in zip(top_words, top_freqs):
        print(f"    {word:<20s} {int(freq):,}")

    plt.figure(figsize=(10, 6))
    plt.bar(top_words, top_freqs)
    plt.xticks(rotation=45)
    plt.tight_layout()

    vis_path = os.path.join(OUTPUTS_DIR, "top_words_visualization.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"\n  Visualization saved to: {vis_path}")

    return (X_train_cv, X_test_cv, cv_vec,
            X_train_tfidf, X_test_tfidf, tfidf_vec)


def step5_train_and_evaluate(X_train_cv, X_test_cv, cv_vec,
                              X_train_tfidf, X_test_tfidf, tfidf_vec,
                              y_train, test_df):
    separator("STEP 5: MODEL TRAINING + EVALUATION")

    y_true = test_df["label"].values

    feature_sets = {
        "CountVectorizer": (X_train_cv, X_test_cv, cv_vec),
        "TF-IDF":          (X_train_tfidf, X_test_tfidf, tfidf_vec),
    }

    results      = {}
    trained      = {}

    for feat_name, (X_tr, X_te, vec) in feature_sets.items():
        for model_name, model in get_models().items():
            combo = f"{model_name} + {feat_name}"
            trained_model = train_model(model, X_tr, y_train, label=feat_name)
            y_pred, y_prob = predict(trained_model, X_te)
            acc, auc, _    = evaluate_model(y_true, y_pred, y_prob, model_name=combo)
            results[combo] = (acc, auc)
            trained[combo] = (trained_model, vec)

    return results, trained, y_true


def step6_compare_and_save(results, trained):
    separator("STEP 6: MODEL COMPARISON & SAVING BEST MODEL")

    best_name = compare_models(results)

    best_model, best_vec = trained[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vec_path   = os.path.join(MODELS_DIR, "best_vectorizer.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(best_vec,   vec_path)

    print(f"\n  Best model saved    : {model_path}")
    print(f"  Best vectorizer saved: {vec_path}")

    return best_name

def step7_sanity_checks():
    separator("STEP 7: SANITY CHECKS")

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vec_path = os.path.join(MODELS_DIR, "best_vectorizer.joblib")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    test_inputs = [
        "I love this",
        "I hate this",
        "This is not good",
        "This is amazing",
        "I just love waiting in traffic for hours"
    ]

    LABEL_MAP = {0: "Negative", 1: "Positive"}

    for text in test_inputs:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = float(model.predict_proba(X)[0][1])
        sentiment = LABEL_MAP.get(int(pred), str(pred))
        print(f"\n  Input: \"{text}\"")
        print(f"  Cleaned: \"{cleaned}\"")
        print(f"  Prediction: {sentiment} (Prob Pos: {prob:.4f})")

def step8_feature_importance(model, vectorizer):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    coefs = model.coef_[0]
    words = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(coefs)
    
    top_negative_indices = sorted_indices[:20]
    top_positive_indices = sorted_indices[-20:][::-1]
    
    top_neg = [(words[i], coefs[i]) for i in top_negative_indices]
    top_pos = [(words[i], coefs[i]) for i in top_positive_indices]
    
    print("\n  Top 20 Positive Words:")
    for w, c in top_pos:
        print(f"    {w:<20} {c:.4f}")
        
    print("\n  Top 20 Negative Words:")
    for w, c in top_neg:
        print(f"    {w:<20} {c:.4f}")
        
    with open(os.path.join(OUTPUTS_DIR, "top_words.txt"), "w", encoding="utf-8") as f:
        f.write("Top 20 Positive Words:\n")
        for w, c in top_pos:
            f.write(f"{w}: {c:.4f}\n")
        f.write("\nTop 20 Negative Words:\n")
        for w, c in top_neg:
            f.write(f"{w}: {c:.4f}\n")

def main():
    print("=" * 60)
    print("  Twitter Sentiment Analysis — Full Pipeline")
    print("=" * 60)

    train_df, test_df = step1_load_data()
    train_df, test_df = step2_preprocess(train_df, test_df)
    step3_eda(train_df)

    (X_train_cv, X_test_cv, cv_vec,
     X_train_tfidf, X_test_tfidf, tfidf_vec) = step4_features(train_df, test_df)

    y_train = train_df["label"].values

    results, trained, y_true = step5_train_and_evaluate(
        X_train_cv, X_test_cv, cv_vec,
        X_train_tfidf, X_test_tfidf, tfidf_vec,
        y_train, test_df
    )

    best_name = step6_compare_and_save(results, trained)
    
    if "Logistic Regression" in best_name:
        best_model, best_vec = trained[best_name]
        step8_feature_importance(best_model, best_vec)

    step7_sanity_checks()

    print(f"\nPipeline complete. Best combination: {best_name}")


if __name__ == "__main__":
    main()