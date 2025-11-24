# src/app.py
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)            # -> /.../ir-rf/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # -> /.../ir-rf
if PROJECT_ROOT not in sys.path:
    # insert at front so it takes precedence
    sys.path.insert(0, PROJECT_ROOT)

import traceback
from typing import List, Tuple
import streamlit as st

# import your modules (they should already work as package-style imports)
from src.hybrid_retrieve import hybrid_search
from src.data_loader import load_cranfield
from src.indexer import load_index
from src.semantic_index import load_semantic_index

# --- helper functions ---
def safe_hybrid_search(q: str, topk: int, lambda_weight: float):
    """Run hybrid_search and capture exceptions to show to user."""
    try:
        st.write(f"Searching for: **{q}** (topk={topk}, λ={lambda_weight:.2f})")
        res = hybrid_search(q, topk=topk, lambda_weight=lambda_weight)
        return res, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, tb

@st.cache_resource
def get_indexes():
    """Load TF-IDF / semantic indices and return them (cached)."""
    # load tfidf index (will build if missing)
    vec, X, docs = load_index()
    # load semantic embeddings (will build if missing)
    emb_matrix, docs2, model = load_semantic_index()
    # ensure alignment
    if len(docs) != len(docs2):
        st.warning("Warning: docs length mismatch between lexical and semantic indexes.")
    return vec, X, docs, emb_matrix, model

# --- UI layout ---
st.set_page_config(layout="wide", page_title="Personalized Query Suggestion — Demo")
st.title("🔍 Personalized Query Suggestion — Demo")

# left column - controls
left, main, right = st.columns([1, 3, 1])
with left:
    st.header("Demo controls")
    topk = st.slider("Top-k to show", min_value=1, max_value=50, value=10)
    lambda_weight = st.slider("Hybrid λ (semantic weight)", min_value=0.0, max_value=1.0, value=0.5)
    rocchio_alpha = st.slider("Rocchio α", 0.0, 2.0, 1.0)
    rocchio_beta = st.slider("Rocchio β", 0.0, 2.0, 0.75)
    st.caption("Tip: click the ✅ Relevant button to mark a doc as relevant (Rocchio feedback).")

with right:
    st.header("Queries in dataset")
    try:
        _, queries_df, _ = load_cranfield()
        # show a few queries
        for i, row in queries_df.head(12).iterrows():
            st.write(f"{row['qid']}: {row['query']}")
    except Exception as e:
        st.write("Could not load sample queries.")

with main:
    query = st.text_input("Enter query", value="", key="query_input")
    search_btn = st.button("Search")

    # diagnostics area
    with st.expander("Diagnostics / logs", expanded=False):
        st.text("Terminal logs will appear here if errors occur.")

    # load cached indexes once
    with st.spinner("Loading indexes... (cached)"):
        try:
            vec, X, docs, emb_matrix, model = get_indexes()
            st.success(f"Loaded indexes: docs={len(docs)}, TFIDF matrix shape={X.shape}")
        except Exception:
            st.error("Error loading indexes. See terminal for traceback.")
            st.text(traceback.format_exc())
            st.stop()

    # handle search action
    if search_btn and query.strip():
        results, error_tb = safe_hybrid_search(query.strip(), topk=topk, lambda_weight=lambda_weight)
        if error_tb:
            st.error("Search failed — see traceback below.")
            st.code(error_tb)
        else:
            # store results in session_state so they persist across reruns
            st.session_state["last_results"] = results
            st.success(f"Found {len(results)} results (showing top {topk}).")

    # If no new search, show previous results if available
    results = st.session_state.get("last_results", None)
    if results is None:
        st.info("No search run yet. Enter a query above and press Search.")
    else:
        # results expected to be a pandas DataFrame (docs.iloc[...] with added score columns)
        try:
            # show query suggestions (if you have a generate_suggestions function)
            try:
                from expand import generate_suggestions
                clicked_texts = []  # no clicks yet
                suggs = generate_suggestions(query.strip(), clicked_texts, k=3) if query.strip() else []
            except Exception:
                suggs = []
            if suggs:
                st.subheader("Query Suggestions (MMR-diversified)")
                cols = st.columns(len(suggs))
                for c, s in zip(cols, suggs):
                    if c.button(s):
                        st.session_state["query_input"] = s
                        st.experimental_rerun()

            st.subheader("Results")
            # results likely a pandas DataFrame with columns id,title,score,tf_score,sem_score
            # show them in a readable form
            for i, row in results.iterrows():
                container = st.container()
                with container:
                    st.markdown(f"**D{row['id']} — {row['title']}**")
                    st.write(row.get("text", "")[:400] + "...")
                    scores = f"score={row.get('score', 0):.4f}  |  tf={row.get('tf_score', 0):.4f}  |  sem={row.get('sem_score', 0):.4f}"
                    st.caption(scores)
                    # controls for feedback
                    cols_fb = st.columns([0.2, 0.2, 1])
                    if cols_fb[0].button("✅ Relevant", key=f"rel_{row['id']}"):
                        # handle Rocchio feedback: for now we just display acknowledgement
                        st.success("Marked as relevant (Rocchio will be applied in next run).")
                    if cols_fb[1].button("Copy ID", key=f"copy_{row['id']}"):
                        st.write(f"Document id {row['id']} copied (use in experiments).")
        except Exception:
            st.error("Error rendering results.")
            st.code(traceback.format_exc())
