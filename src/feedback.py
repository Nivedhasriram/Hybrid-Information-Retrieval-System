# src/feedback.py
import numpy as np
from sklearn.preprocessing import normalize
from .indexer import load_index
from .semantic_index import load_semantic_index
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import load_cranfield

ALPHA = 1.0
BETA = 0.8
GAMMA = 0.0

def rocchio_tfidf(query_vec, rel_doc_vecs, nonrel_doc_vecs=None, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    # query_vec: 1D array; rel_doc_vecs: 2D
    q = query_vec.toarray().flatten() if hasattr(query_vec, "toarray") else np.array(query_vec).flatten()
    new_q = alpha * q
    if rel_doc_vecs is not None and len(rel_doc_vecs)>0:
        new_q += beta * np.mean(rel_doc_vecs, axis=0)
    if nonrel_doc_vecs is not None and len(nonrel_doc_vecs)>0:
        new_q -= gamma * np.mean(nonrel_doc_vecs, axis=0)
    return new_q

def rocchio_emb(query_emb, rel_embs, nonrel_embs=None, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    q = query_emb.flatten()
    new_q = alpha * q
    if rel_embs is not None and len(rel_embs)>0:
        new_q = new_q + beta * np.mean(rel_embs, axis=0)
    if nonrel_embs is not None and len(nonrel_embs)>0:
        new_q = new_q - gamma * np.mean(nonrel_embs, axis=0)
    # normalize
    new_q = new_q / (np.linalg.norm(new_q)+1e-9)
    return new_q

def apply_feedback_for_query(qtext, clicked_doc_ids, topk=100, lambda_weight=0.5):
    # load indices
    vectorizer, X, docs = load_index()
    emb_matrix, docs2, model = load_semantic_index()
    qv = vectorizer.transform([qtext])
    q_emb = model.encode([qtext], convert_to_numpy=True)
    # find doc vectors for clicked docs
    doc_index_map = {d: i for i,d in enumerate(docs['id'])}
    rel_idxs = [doc_index_map[did] for did in clicked_doc_ids if did in doc_index_map]
    rel_tfidf_vectors = X[rel_idxs].toarray() if len(rel_idxs)>0 else None
    rel_embs = emb_matrix[rel_idxs] if len(rel_idxs)>0 else None
    # rocchio
    qv_new = rocchio_tfidf(qv, rel_tfidf_vectors)
    q_emb_new = rocchio_emb(q_emb, rel_embs)
    # compute updated hybrid score
    tf_scores = cosine_similarity(qv_new.reshape(1,-1), X).flatten()
    sem_scores = cosine_similarity(q_emb_new.reshape(1,-1), emb_matrix).flatten()
    scores = lambda_weight * sem_scores + (1-lambda_weight) * tf_scores
    idx = (-scores).argsort()[:topk]
    return docs.iloc[idx].assign(score=scores[idx], tf_score=tf_scores[idx], sem_score=sem_scores[idx])
