# src/expand.py 
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# lazy import to avoid heavy dependency at top-level
def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers")
    return SentenceTransformer("all-MiniLM-L6-v2")

STOPWORDS = set([
    "the","a","an","in","on","of","and","or","is","are","to","for","with","that","this","it","as","by","from"
])  # expand with NLTK stopwords if available

def _tokenize_words(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]
    return tokens

def _extract_candidate_phrases(texts, min_freq=1, max_ngram=3):
    # produce ngrams and count frequencies
    counter = Counter()
    for t in texts:
        tokens = _tokenize_words(t)
        for n in range(1, max_ngram+1):
            for i in range(len(tokens)-n+1):
                ng = tokens[i:i+n]
                # require at least one non-query-like word and length>1 char
                phrase = " ".join(ng)
                if len(phrase) < 3: continue
                counter[phrase] += 1
    # keep frequent candidates
    candidates = [p for p,c in counter.items() if c >= min_freq]
    # sort by freq desc
    candidates.sort(key=lambda x: -counter[x])
    return candidates, counter

def _candidate_tfidf_scores(candidates, query):
    if not candidates:
        return {}
    vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='word', min_df=1)
    # Fit on candidates + query to get relative tfidf
    docs = candidates + [query]
    X = vectorizer.fit_transform(docs)
    # query is last row
    qv = X[-1]
    cand_matrix = X[:-1]
    # compute cosine similarity between each candidate and the query (using TFIDF vectors)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(cand_matrix, qv).flatten()
    return {candidates[i]: float(sims[i]) for i in range(len(candidates))}

def generate_suggestions(query, seed_texts, k=3, mmr_lambda=0.7, min_freq=1, max_ngram=3):
    """
    query: str
    seed_texts: list[str]  (clicked doc texts or top doc texts)
    returns: list[str] (k suggested query reformulations)
    """
    # 1) extract meaningful candidate phrases
    candidates, counter = _extract_candidate_phrases(seed_texts, min_freq=min_freq, max_ngram=max_ngram)
    if not candidates:
        return []

    # 2) quick filter: remove candidates that are identical to the original query or trivial
    qnorm = " ".join(_tokenize_words(query))
    candidates = [c for c in candidates if c != qnorm and len(c.split()) <= 4]

    # 3) score candidates by TF-IDF similarity to query
    tfidf_scores = _candidate_tfidf_scores(candidates, query)  # 0..1

    # 4) compute SBERT embeddings
    model = _get_sentence_transformer()
    cand_embs = model.encode(candidates, convert_to_numpy=True)
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    # 5) MMR selection
    selected = []
    selected_idxs = []
    cand_idxs = list(range(len(candidates)))
    # Precompute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_q_c = cosine_similarity([q_emb], cand_embs).flatten()  # relevance
    sim_c_c = cosine_similarity(cand_embs, cand_embs)

    # combine TF-IDF into initial relevance ranking by multiplying (optional)
    # We'll use a combined_relevance = alpha * sim_q_c + (1-alpha)*tfidf_score
    alpha = 0.7
    combined_relevance = []
    for i, c in enumerate(candidates):
        tf = tfidf_scores.get(c, 0.0)
        combined_relevance.append(alpha * sim_q_c[i] + (1-alpha) * tf)

    # pick first = highest combined_relevance
    if len(candidates) > 0:
        first_idx = int(np.argmax(combined_relevance))
        selected_idxs.append(first_idx)
        selected.append(candidates[first_idx])

    while len(selected) < k and len(selected) < len(candidates):
        mmr_scores = []
        for i in cand_idxs:
            if i in selected_idxs:
                mmr_scores.append(-np.inf)
                continue
            # relevance
            rel = combined_relevance[i]
            # redundancy = max similarity to any selected
            red = max(sim_c_c[i, j] for j in selected_idxs) if selected_idxs else 0.0
            score = mmr_lambda * rel - (1 - mmr_lambda) * red
            mmr_scores.append(score)
        next_idx = int(np.argmax(mmr_scores))
        selected_idxs.append(next_idx)
        selected.append(candidates[next_idx])

    return selected
