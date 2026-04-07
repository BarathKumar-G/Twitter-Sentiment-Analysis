import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

VECTORIZER_PARAMS = dict(
    max_features=50000,
    min_df=2,
    ngram_range=(1, 2),
    stop_words="english",
)


def build_count_features(train_texts, test_texts):
    vectorizer = CountVectorizer(**VECTORIZER_PARAMS)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  CountVectorizer - vocabulary size: {vocab_size:,}")
    print(f"  Train matrix: {X_train.shape}, Test matrix: {X_test.shape}")

    return X_train, X_test, vectorizer


def build_tfidf_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"  TF-IDF Vectorizer - vocabulary size: {vocab_size:,}")
    print(f"  Train matrix: {X_train.shape}, Test matrix: {X_test.shape}")

    return X_train, X_test, vectorizer


def get_top_words(vectorizer, X, n=20):
    sums = np.array(X.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    top_indices = sums.argsort()[::-1][:n]
    return [words[i] for i in top_indices]