# Hybrid Information Retrieval System

A hybrid search engine built using **TF-IDF**, **Semantic Search (Sentence-BERT)**, and **Rocchio relevance feedback**, demonstrated with an interactive **Streamlit UI**.  
The system uses the **Cranfield dataset** (1400 documents, 225 queries) and supports end-to-end retrieval, ranking, personalization, and evaluation.

## Features

### Hybrid Retrieval
- **TF-IDF lexical matching**
- **Semantic similarity** with Sentence-Transformer (`all-MiniLM-L6-v2`)
- Adjustable hybrid score:

score = λ * semantic + (1 − λ) * tfidf

### Personalized Search (Rocchio)
- Relevance feedback updates the query vector:

q_new = α * q + β * mean(relevant_docs)

### Streamlit User Interface
- Top-k results slider  
- Hybrid λ slider  
- Rocchio α & β sliders  
- “Mark Relevant” button to improve results  
- Diagnostics panel  
- Sidebar showing sample Cranfield queries

### Dataset

Uses the Cranfield Collection:
1400 documents
225 queries
Standard qrels relevance judgments
Converted to JSONL for efficient indexing.

### Evaluation
- Supports **Precision@k** and **nDCG@k** using qrels.

### Demo Video
- https://drive.google.com/file/d/1241wxDvx3g5_ouVi13EHPepzPKHfwtsD/view?usp=sharing

## Run the App
streamlit run src/app.py
The app opens at: http://localhost:8501

## Setup

```bash
git clone <repo_url>
cd ir-rf

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt



