import os
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import load_data, preprocess
from src.features       import build_count_features, build_tfidf_features
from src.model          import train_model, predict
from src.evaluation     import evaluate_model, show_sample_predictions, error_analysis


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_data.csv")

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
    dist = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    for label, count in dist.items():
        name = "Negative" if label == 0 else "Positive"
        pct  = count / total * 100
        print(f"    {name} ({label}): {count:,}  ({pct:.1f}%)")

    train_df["tweet_len"] = train_df["clean_text"].str.split().str.len()
    print("\n  Tweet Length Stats:")
    print(f"    min   : {np.min(train_df['tweet_len']):.1f}")
    print(f"    max   : {np.max(train_df['tweet_len']):.1f}")
    print(f"    mean  : {np.mean(train_df['tweet_len']):.1f}")
    print(f"    median: {np.median(train_df['tweet_len']):.1f}")


def step4_features(train_df, test_df):
    separator("STEP 4: FEATURE EXTRACTION")

    train_texts = train_df["clean_text"].values
    test_texts  = test_df["clean_text"].values

    print("\n  [CountVectorizer]")
    X_train_cv, X_test_cv, cv_vec = build_count_features(train_texts, test_texts)

    print("\n  [TF-IDF]")
    X_train_tfidf, X_test_tfidf, tfidf_vec = build_tfidf_features(train_texts, test_texts)

    # 🔥 Correct top words (based on vectorizer, NOT raw text)
    print("\n  Top 20 Words (after vectorization):")

    sums = np.array(X_train_cv.sum(axis=0)).flatten()
    words = cv_vec.get_feature_names_out()

    top_indices = sums.argsort()[::-1][:20]

    top_words = [words[i] for i in top_indices]
    top_freqs = [sums[i] for i in top_indices]

    for word, freq in zip(top_words, top_freqs):
        print(f"    {word:<20s} {int(freq):,}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(top_words, top_freqs)
    plt.xticks(rotation=45)
    plt.tight_layout()

    vis_path = os.path.join(BASE_DIR, 'top_words_visualization.png')
    plt.savefig(vis_path)
    plt.close()

    print(f"\n  Visualization saved to: {vis_path}")

    return (X_train_cv, X_test_cv, cv_vec,
            X_train_tfidf, X_test_tfidf, tfidf_vec)


def step5_train(X_train_cv, y_train, X_train_tfidf):
    separator("STEP 5: MODEL TRAINING")

    model_cv    = train_model(X_train_cv,    y_train, label="CountVectorizer")
    model_tfidf = train_model(X_train_tfidf, y_train, label="TF-IDF")

    return model_cv, model_tfidf


def step6_evaluate(model_cv, model_tfidf,
                   X_test_cv, X_test_tfidf,
                   test_df):
    separator("STEP 6: EVALUATION")

    y_true = test_df["label"].values

    # CountVectorizer
    y_pred_cv, y_prob_cv = predict(model_cv, X_test_cv)
    acc_cv, cm_cv = evaluate_model(
        y_true=y_true,
        y_pred=y_pred_cv,
        y_prob=y_prob_cv,
        model_name="Logistic Regression + CountVectorizer"
    )

    show_sample_predictions(
        texts=test_df["text"].values,
        y_true=y_true,
        y_pred=y_pred_cv,
        n=8,
        model_name="CountVectorizer"
    )

    # TF-IDF
    y_pred_tfidf, y_prob_tfidf = predict(model_tfidf, X_test_tfidf)
    acc_tfidf, cm_tfidf = evaluate_model(
        y_true=y_true,
        y_pred=y_pred_tfidf,
        y_prob=y_prob_tfidf,
        model_name="Logistic Regression + TF-IDF"
    )

    show_sample_predictions(
        texts=test_df["text"].values,
        y_true=y_true,
        y_pred=y_pred_tfidf,
        n=8,
        model_name="TF-IDF"
    )

    separator("MODEL COMPARISON")
    print(f"\n  CountVectorizer Accuracy : {acc_cv*100:.2f}%")
    print(f"  TF-IDF Accuracy          : {acc_tfidf*100:.2f}%")

    return y_true, y_pred_cv, y_pred_tfidf


def step7_error_analysis(test_df, y_true, y_pred_cv, y_pred_tfidf):
    separator("STEP 7: ERROR ANALYSIS")

    texts = test_df["text"].values

    error_analysis(texts, y_true, y_pred_cv,    n=5, model_name="CountVectorizer")
    error_analysis(texts, y_true, y_pred_tfidf, n=5, model_name="TF-IDF")


def main():
    print("=" * 60)
    print("  Twitter Sentiment Analysis — Baseline Pipeline")
    print("=" * 60)

    train_df, test_df = step1_load_data()
    train_df, test_df = step2_preprocess(train_df, test_df)
    step3_eda(train_df)

    (X_train_cv, X_test_cv, cv_vec,
     X_train_tfidf, X_test_tfidf, tfidf_vec) = step4_features(train_df, test_df)

    y_train = train_df["label"].values
    model_cv, model_tfidf = step5_train(X_train_cv, y_train, X_train_tfidf)

    y_true, y_pred_cv, y_pred_tfidf = step6_evaluate(
        model_cv, model_tfidf,
        X_test_cv, X_test_tfidf,
        test_df
    )

    step7_error_analysis(test_df, y_true, y_pred_cv, y_pred_tfidf)

    print("\nBaseline NLP pipeline working successfully")


if __name__ == "__main__":
    main()