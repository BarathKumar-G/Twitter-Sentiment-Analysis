"""
features.py - Feature extraction for Twitter Sentiment Analysis.

Implements two vectorisation strategies:
  1. CountVectorizer  — raw term frequencies (bag-of-words)
  2. TF-IDF Vectorizer — term frequency × inverse document frequency

Both vectorisers are *fit only on training data* and then used to transform
test data, preventing data leakage.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ─── Shared Vectoriser Settings ───────────────────────────────────────────────

VECTORIZER_PARAMS = dict(
    max_features=50_000,   # Cap vocabulary size for RAM efficiency
    min_df=2,              # Ignore terms appearing in fewer than 2 docs
    ngram_range=(1, 2),    # Unigrams + bigrams
    stop_words="english",  # Remove common English stopwords
)


# ─── CountVectorizer ──────────────────────────────────────────────────────────

def build_count_features(train_texts, test_texts):
    """
    Fit a CountVectorizer on training text and transform both splits.

    Args:
        train_texts: Iterable of cleaned training tweet strings.
        test_texts:  Iterable of cleaned test tweet strings.

    Returns:
        (X_train_count, X_test_count, count_vectorizer)
    """
    vectorizer = CountVectorizer(**VECTORIZER_PARAMS)

    X_train = vectorizer.fit_transform(train_texts)   # Fit + transform train
    X_test  = vectorizer.transform(test_texts)        # Transform only

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  CountVectorizer — vocabulary size: {vocab_size:,}")
    print(f"  Train matrix: {X_train.shape}, Test matrix: {X_test.shape}")

    return X_train, X_test, vectorizer


# ─── TF-IDF Vectorizer ────────────────────────────────────────────────────────

def build_tfidf_features(train_texts, test_texts):
    """
    Fit a TfidfVectorizer on training text and transform both splits.

    Args:
        train_texts: Iterable of cleaned training tweet strings.
        test_texts:  Iterable of cleaned test tweet strings.

    Returns:
        (X_train_tfidf, X_test_tfidf, tfidf_vectorizer)
    """
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)

    X_train = vectorizer.fit_transform(train_texts)   # Fit + transform train
    X_test  = vectorizer.transform(test_texts)        # Transform only

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  TF-IDF Vectorizer — vocabulary size: {vocab_size:,}")
    print(f"  Train matrix: {X_train.shape}, Test matrix: {X_test.shape}")

    return X_train, X_test, vectorizer


# ─── Top Words Helper ─────────────────────────────────────────────────────────

import numpy as np

def get_top_words(vectorizer, X, n=20):
    sums = np.array(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    top_indices = sums.argsort()[::-1][:n]
    return [words[i] for i in top_indices]