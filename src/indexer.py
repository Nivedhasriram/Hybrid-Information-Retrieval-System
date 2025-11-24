# src/indexer.py 
import joblib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .data_loader import load_cranfield
from .preprocess import normalize_text

# Use these constants; change if you want another folder
DATA_DIR = "data"
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.joblib")
VEC_MATRIX_PATH = os.path.join(DATA_DIR, "tfidf_matrix.joblib")
DOCS_PATH = os.path.join(DATA_DIR, "docs.csv")

def build_index(data_dir=DATA_DIR, min_df=1, ngram_range=(1,2)):
    corpus_df, _, _ = load_cranfield()

    # normalize text
    corpus_df['text_norm'] = corpus_df['text'].fillna("").astype(str).apply(normalize_text)
    # quick stats
    non_empty = (corpus_df['text_norm'].str.strip() != "").sum()
    total = len(corpus_df)
    print(f"Building index for {total} docs ({non_empty} with non-empty text_norm)")

    docs = corpus_df.copy().reset_index(drop=True)

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words="english")
    X = vectorizer.fit_transform(docs['text_norm'])

    # ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    joblib.dump(vectorizer, TFIDF_PATH)
    joblib.dump(X, VEC_MATRIX_PATH)
    docs.to_csv(DOCS_PATH, index=False, encoding="utf8")

    print("Index built:", X.shape)
    return vectorizer, X, docs

def load_index(data_dir=DATA_DIR):
    """Load TF-IDF index if it exists, otherwise build it."""
    try:
        vectorizer = joblib.load(TFIDF_PATH)
        X = joblib.load(VEC_MATRIX_PATH)
        docs = pd.read_csv(DOCS_PATH, encoding="utf8")
        print("Loaded existing index:", X.shape)
    except FileNotFoundError:
        print("Index files missing, building new index...")
        vectorizer, X, docs = build_index(data_dir)
    return vectorizer, X, docs

if __name__=="__main__":
    build_index()
