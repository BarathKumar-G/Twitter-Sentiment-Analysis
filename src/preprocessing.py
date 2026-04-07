import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

_lemmatizer = WordNetLemmatizer()

def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

TEXT_ALIASES = ["sentence", "text", "tweet", "content", "message"]
LABEL_ALIASES = ["sentiment", "label", "target", "class", "polarity"]


def detect_columns(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}

    text_col = None
    for alias in TEXT_ALIASES:
        if alias in cols_lower:
            text_col = cols_lower[alias]
            break

    label_col = None
    for alias in LABEL_ALIASES:
        if alias in cols_lower:
            label_col = cols_lower[alias]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns. Available columns: {list(df.columns)}\n"
            f"Text aliases tried: {TEXT_ALIASES}\n"
            f"Label aliases tried: {LABEL_ALIASES}"
        )

    return text_col, label_col


def load_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
    print(f"  Raw shape: {df.shape}")

    text_col, label_col = detect_columns(df)
    print(f"  Detected columns - text: '{text_col}', label: '{label_col}'")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing values.")

    df["label"] = df["label"].astype(int)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  Sampled {sample_size} rows from training data.")

    print(f"  Final shape: {df.shape}")
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"&amp;?", " and ", text)
    text = re.sub(r"&quot;?", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\bamp\b", " and ", text)
    text = re.sub(r"\bquot\b", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    tokens = text.split()
    pos_tags = nltk.pos_tag(tokens)
    text = " ".join(_lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in pos_tags)
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df)} empty rows after cleaning.")

    return df
