# src/data_loader.py

from pathlib import Path
import json
import re
from collections import defaultdict
import pandas as pd

BASE = Path("data/cranfield")
RAW = Path("data/cranfield_raw")

CORPUS_JSONL = BASE / "corpus.jsonl"
QUERIES_JSONL = BASE / "queries.jsonl"
QRELS_JSONL = BASE / "qrels.jsonl"

# raw names we might find
RAW_DOC_CANDIDATES = ["cran.all.1400", "cran.all.1400.xml", "cran.all", "cran.all.txt"]
RAW_QRY_CANDIDATES = ["cran.qry", "cran.qry.xml", "cran.qry.txt", "cran.queries"]
RAW_QREL_CANDIDATES = ["cranqrel", "cran.qrel", "cranqrel.txt"]

def read_first_existing(path_candidates, parent=RAW):
    for name in path_candidates:
        p = parent / name
        if p.exists():
            return p.read_text(encoding="utf8", errors="ignore"), p
    return "", None

def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed lines, but continue
                continue
    return out

def parse_raw_docs_from_text(text: str):
    """
    Parse either TREC-style .I/.T/.W or XML-like <doc> ... </doc> formats.
    Returns list of dicts: {'id':..., 'title':..., 'text':...}
    """
    docs = []

    if not text:
        return docs

    # Option A: TREC-style with ".I " separators
    if ".I " in text:
        parts = re.split(r"\n\.I\s+", text)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            header_rest = p.split("\n", 1)
            docid = header_rest[0].strip().split()[0]
            body = header_rest[1] if len(header_rest) > 1 else ""
            title = ""
            wtext = ""
            tmatch = re.search(r"\.T\s*(.*?)\n(?=(?:\.[A-Z]\s)|$)", p, re.S)
            wmatch = re.search(r"\.W\s*(.*)", p, re.S)
            if tmatch:
                title = re.sub(r"\s+", " ", tmatch.group(1)).strip()
            if wmatch:
                wtext = re.sub(r"\s+", " ", wmatch.group(1)).strip()
            docs.append({"id": str(docid), "title": title, "text": wtext})
        if docs:
            return docs

    # Option B: XML-like <doc> ... </doc>
    if "<doc" in text.lower() or "<docno" in text.lower():
        for m in re.finditer(r"(?is)<doc[^>]*>(.*?)</doc>", text):
            inner = m.group(1)
            docno_m = re.search(r"(?is)<docno[^>]*>(.*?)</docno>", inner)
            docno = docno_m.group(1).strip() if docno_m else None
            title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", inner)
            title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
            text_m = re.search(r"(?is)<text[^>]*>(.*?)</text>", inner)
            body = re.sub(r"\s+", " ", text_m.group(1)).strip() if text_m else ""
            if not docno:
                lines = re.sub(r"(?is)<[^>]+>", " ", inner).strip().splitlines()
                if lines and re.match(r"^\d+$", lines[0].strip()):
                    docno = lines[0].strip()
            if not docno:
                continue
            docs.append({"id": str(docno), "title": title, "text": body})
        if docs:
            return docs

    # fallback: try splitting by "<doc" roughly
    return docs

def parse_raw_queries_from_text(text: str):
    """
    Parse queries in XML <top><num>...<title>... format or TREC .I/.W style
    Returns list of {'qid':..., 'query':...}
    """
    queries = []
    if not text:
        return queries

    # TREC-style .I .W
    if ".I " in text:
        parts = re.split(r"\n\.I\s+", text)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            header_rest = p.split("\n", 1)
            qid = header_rest[0].strip().split()[0]
            wmatch = re.search(r"\.W\s*(.*)", p, re.S)
            qstr = re.sub(r"\s+", " ", wmatch.group(1)).strip() if wmatch else (header_rest[1].strip() if len(header_rest) > 1 else "")
            queries.append({"qid": "Q" + str(qid) if not str(qid).startswith("Q") else str(qid), "query": qstr})
        if queries:
            return queries

    # XML-like <top> blocks
    if "<top" in text.lower() or "<num" in text.lower() or "<title" in text.lower():
        tops = re.findall(r"(?is)<top[^>]*>(.*?)</top>", text)
        for t in tops:
            num_m = re.search(r"(?is)<num[^>]*>(.*?)</num>", t)
            title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", t)
            if num_m and title_m:
                qid_raw = re.sub(r"\D", "", num_m.group(1))
                qid = f"Q{qid_raw}" if qid_raw else num_m.group(1).strip()
                qtext = re.sub(r"\s+", " ", title_m.group(1)).strip()
                queries.append({"qid": qid, "query": qtext})
        if queries:
            return queries

    # fallback: split lines looking for leading numbers
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if re.match(r"^\d+\b", lines[i]):
            qid = re.match(r"^(\d+)", lines[i]).group(1)
            i += 1
            qlines = []
            while i < len(lines) and not re.match(r"^\d+\b", lines[i]):
                qlines.append(lines[i]); i += 1
            queries.append({"qid": f"Q{qid}", "query": " ".join(qlines)})
        else:
            i += 1
    return queries

def parse_raw_qrels_from_text(text: str):
    """
    Parse qrels where each line is like:
      qid  0  docid  relevance
    or JSON lines of either aggregated or single entries.
    Returns list of entries or aggregated dict.
    """
    entries = []
    if not text:
        return entries
    # If it's JSONL-like lines, try to parse JSON lines and return them
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        # try JSON
        try:
            j = json.loads(line)
            # normalize keys
            entries.append(j)
            continue
        except Exception:
            pass
        # try whitespace-separated numbers
        parts = re.split(r"\s+", line)
        if len(parts) >= 3:
            qid = parts[0]
            docid = parts[2]
            score = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
            entries.append({"qid": f"Q{qid}" if not str(qid).startswith("Q") else str(qid), "docid": str(docid), "score": score})
    return entries

def build_qrels_map_from_entries(entries):
    """
    Accepts a list of entries where each entry might be:
     - {'qid': 'Q1', 'relevant': ['1','5']}
     - {'qid':'Q1', 'docid':'1', 'score':1}
     - {'qid':'1', 'docid':'1', 'score':1}
    Returns dict qid -> [docid, ...]
    """
    qrels_map = defaultdict(list)
    for r in entries:
        if not isinstance(r, dict):
            continue
        qid = str(r.get("qid") or r.get("QID") or r.get("num") or r.get("query") or "").strip()
        if qid and not qid.startswith("Q"):
            # if converter used numeric qid, standardize to Q#
            if re.match(r"^\d+$", qid):
                qid = f"Q{qid}"
        # case: aggregated
        if "relevant" in r and isinstance(r["relevant"], (list, tuple)):
            for d in r["relevant"]:
                qrels_map[qid].append(str(d))
            continue
        # case: docid per-line
        doc = r.get("docid") or r.get("doc") or r.get("document") or r.get("DOCID")
        if doc:
            qrels_map[qid].append(str(doc))
            continue
        # if a json line had 'qid' and 'docid' in different naming, try to extract numeric keys
        # fallback: ignore
    # remove duplicates and ensure list
    out = {k: list(dict.fromkeys(v)) for k, v in qrels_map.items()}
    return out

def load_cranfield():
    """
    Main loader entrypoint.
    Returns: (corpus_df, queries_df, qrels_map)
      - corpus_df: pd.DataFrame with columns ['id','title','text']
      - queries_df: pd.DataFrame with columns ['qid','query']
      - qrels_map: dict mapping qid -> [docid, ...]
    """
    # --- load corpus ---
    corpus_list = load_jsonl(CORPUS_JSONL)
    if not corpus_list:
        # try raw doc candidates
        text, fpath = read_first_existing(RAW_DOC_CANDIDATES, parent=RAW)
        if text:
            corpus_list = parse_raw_docs_from_text(text)
        else:
            # fallback: empty
            corpus_list = []

    # Normalize corpus into DataFrame
    corpus_df = pd.DataFrame(corpus_list)
    if "id" not in corpus_df.columns:
        corpus_df.rename(columns={corpus_df.columns[0]: "id"}, inplace=True)
    # ensure required columns
    for c in ("id", "title", "text"):
        if c not in corpus_df.columns:
            corpus_df[c] = ""

    # ensure id is string and reset index
    corpus_df["id"] = corpus_df["id"].astype(str)
    corpus_df = corpus_df[["id", "title", "text"]].reset_index(drop=True)

    # --- load queries ---
    queries_list = load_jsonl(QUERIES_JSONL)
    if not queries_list:
        text, qpath = read_first_existing(RAW_QRY_CANDIDATES, parent=RAW)
        if text:
            queries_list = parse_raw_queries_from_text(text)
        else:
            queries_list = []

    queries_df = pd.DataFrame(queries_list)
    if queries_df.empty:
        # ensure a consistent empty dataframe
        queries_df = pd.DataFrame(columns=["qid", "query"])
    else:
        # normalize qid column
        if "qid" not in queries_df.columns and "id" in queries_df.columns:
            queries_df = queries_df.rename(columns={"id": "qid"})
        queries_df["qid"] = queries_df["qid"].astype(str)
        queries_df = queries_df[["qid", "query"]]

    # --- load qrels ---
    qrels_list = load_jsonl(QRELS_JSONL)
    if not qrels_list:
        qtext, qrel_path = read_first_existing(RAW_QREL_CANDIDATES, parent=RAW)
        if qtext:
            qrels_entries = parse_raw_qrels_from_text(qtext)
            qrels_map = build_qrels_map_from_entries(qrels_entries)
        else:
            qrels_map = {}
    else:
        # If load_jsonl returned parsed JSON objects, normalize them into entries and aggregate
        qrels_map = build_qrels_map_from_entries(qrels_list)

    # if queries are empty but qrels exist, we can still populate queries from qrels keys
    if queries_df.empty and qrels_map:
        queries_df = pd.DataFrame([{"qid": k, "query": ""} for k in sorted(qrels_map.keys())])

    # Final normalization: ensure keys are strings in qrels_map
    qrels_map = {str(k): [str(x) for x in v] for k, v in qrels_map.items()}

    return corpus_df, queries_df, qrels_map


# Convenience alias for other modules that expected `load_cranfield`
if __name__ == "__main__":
    c, q, r = load_cranfield()
    print("Loaded corpus:", len(c), "queries:", len(q), "qrels:", len(r))
