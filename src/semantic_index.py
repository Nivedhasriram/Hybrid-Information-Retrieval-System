# src/semantic_index.py
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .data_loader import load_cranfield
from .indexer import build_index
import os

EMB_PATH = "data/doc_embeddings.npy"
EMB_MODEL = "all-MiniLM-L6-v2"
DOCS_PATH = "data/docs.csv"

def build_semantic_index(data_dir="data", model_name=EMB_MODEL):
    # ensure docs exist
    if not os.path.exists(DOCS_PATH):
        build_index(data_dir)
    docs = pd.read_csv(DOCS_PATH)
    model = SentenceTransformer(model_name)
    texts = (docs['title'].fillna("") + ". " + docs['text']).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, embeddings)
    print("Embeddings saved:", embeddings.shape)
    return embeddings, docs, model

def load_semantic_index():
    embeddings = np.load(EMB_PATH)
    docs = pd.read_csv(DOCS_PATH)
    model = SentenceTransformer(EMB_MODEL)
    return embeddings, docs, model

if __name__=="__main__":
    build_semantic_index()
