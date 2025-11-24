# src/retrieve.py
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from .data_loader import load_cranfield
from .preprocess import normalize_text
from .indexer import TFIDF_PATH, VEC_MATRIX_PATH, DOCS_PATH, build_index, load_index

def load_index():
    try:
        vectorizer = joblib.load(TFIDF_PATH)
        X = joblib.load(VEC_MATRIX_PATH)
        docs = pd.read_csv(DOCS_PATH)
    except:
        vectorizer, X, docs = build_index()
    return vectorizer, X, docs

def tfidf_search(query, topk=10):
    vectorizer, X, docs = load_index()
    qnorm = normalize_text(query)
    qv = vectorizer.transform([qnorm])
    scores = cosine_similarity(qv, X).flatten()
    idx = (-scores).argsort()[:topk]
    return docs.iloc[idx].assign(score=scores[idx])

if __name__=="__main__":
    _, queries, qrels = load_cranfield()
    res = tfidf_search("text preprocessing stemming", topk=5)
    print(res[['id','title','score']])
