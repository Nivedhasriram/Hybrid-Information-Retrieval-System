# src/hybrid_retrieve.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.indexer import load_index, build_index
from src.semantic_index import load_semantic_index, build_semantic_index
from src.preprocess import normalize_text
from sentence_transformers import SentenceTransformer

LAMBDA = 0.5  # hybrid weight

def hybrid_search(query, topk=10, lambda_weight=LAMBDA, use_build_if_missing=True):
    # load tfidf
    try:
        vectorizer, X, docs = load_index()
    except:
        vectorizer, X, docs = build_index()
    # load semantic embeddings + model
    try:
        emb_matrix, docs2, model = load_semantic_index()
    except:
        emb_matrix, docs2, model = build_semantic_index()
    # ensure docs alignment
    assert len(docs)==len(docs2)
    # TF-IDF scores
    qv = vectorizer.transform([normalize_text(query)])
    tf_scores = cosine_similarity(qv, X).flatten()
    # semantic scores
    q_emb = model.encode([query], convert_to_numpy=True)
    sem_scores = cosine_similarity(q_emb, emb_matrix).flatten()
    # hybrid
    scores = lambda_weight * sem_scores + (1-lambda_weight) * tf_scores
    idx = (-scores).argsort()[:topk]
    return docs.iloc[idx].assign(score=scores[idx], tf_score=tf_scores[idx], sem_score=sem_scores[idx])
