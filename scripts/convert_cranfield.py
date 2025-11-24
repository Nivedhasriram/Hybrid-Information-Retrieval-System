#!/usr/bin/env python3
# scripts/convert_cranfield.py
from pathlib import Path
import re
import json

RAW_DIR = Path("data/cranfield_raw")
OUT_DIR = Path("data/cranfield")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# candidates for raw doc filename (includes your uploaded path variant)
DOC_CANDIDATES = [
    "cran.all.1400",
    "cran.all.1400.xml",
    "cran.all",
    "cran.all.txt"
]
QRY_CANDIDATES = [
    "cran.qry",
    "cran.qry.txt",
    "cran.queries"
]
QREL_CANDIDATES = [
    "cranqrel",
    "cran.qrel",
    "cranqrel.txt"
]

def read_first_existing(candidates):
    for name in candidates:
        p = RAW_DIR / name
        if p.exists():
            return p.read_text(encoding="utf8", errors="ignore"), name
    return "", None

def parse_qrels_text(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: 
            continue
        parts = line.split()
        # expected: qid  0  docid  relevance  (or similar)
        if len(parts) >= 3:
            qid = parts[0]
            docid = parts[2]
            try:
                score = int(parts[3]) if len(parts) > 3 else 1
            except:
                score = 1
            out.append({"qid": qid, "docid": docid, "score": score})
    return out

def parse_trec_style(text):
    # TREC-style: .I <id> .T <title> .A <author> .W <text>
    if ".I " not in text:
        return []
    docs = []
    parts = re.split(r"\n\.I\s+", text)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # first token is id
        header_rest = p.split("\n",1)
        header = header_rest[0].strip().split()[0]
        body = header_rest[1] if len(header_rest) > 1 else ""
        title = ""
        wtext = ""
        tmatch = re.search(r"\.T\s*(.*?)\n(?=(?:\.[A-Z]\s)|$)", p, re.S)
        wmatch = re.search(r"\.W\s*(.*)", p, re.S)
        if tmatch:
            title = re.sub(r"\s+", " ", tmatch.group(1)).strip()
        if wmatch:
            wtext = re.sub(r"\s+", " ", wmatch.group(1)).strip()
        docs.append({"id": header, "title": title, "text": wtext})
    return docs

def parse_xml_like(text):
    # XML-like: <doc> ... <docno>...</docno> <title>...</title> <text>...</text> </doc>
    if "<doc" not in text and "<DOC" not in text and "<docno" not in text.lower():
        return []
    docs = []
    # find all doc blocks
    for m in re.finditer(r"(?is)<doc[^>]*>(.*?)</doc>", text):
        inner = m.group(1)
        docno_m = re.search(r"(?is)<docno[^>]*>(.*?)</docno>", inner)
        docno = docno_m.group(1).strip() if docno_m else None
        title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", inner)
        title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
        text_m = re.search(r"(?is)<text[^>]*>(.*?)</text>", inner)
        body = re.sub(r"\s+", " ", text_m.group(1)).strip() if text_m else ""
        # fallback: if docno missing, try first numeric line
        if not docno:
            lines = re.sub(r"(?is)<[^>]+>", " ", inner).strip().splitlines()
            if lines and re.match(r"^\d+$", lines[0].strip()):
                docno = lines[0].strip()
        if not docno:
            # skip if we can't get an id
            continue
        docs.append({"id": docno, "title": title, "text": body})
    return docs

def parse_queries_text(text):
    if not text:
        return []
    if ".I " in text:
        qs = []
        parts = re.split(r"\n\.I\s+", text)
        for p in parts:
            p = p.strip()
            if not p: continue
            header_rest = p.split("\n",1)
            qid = header_rest[0].strip().split()[0]
            wmatch = re.search(r"\.W\s*(.*)", p, re.S)
            qstr = re.sub(r"\s+", " ", wmatch.group(1)).strip() if wmatch else (header_rest[1].strip() if len(header_rest)>1 else "")
            qs.append({"qid": qid, "query": qstr})
        return qs
    # fallback: treat lines starting with number as qid
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    qs = []
    i = 0
    while i < len(lines):
        if re.match(r"^\d+\b", lines[i]):
            qid = lines[i].split()[0]
            i += 1
            qlines = []
            while i < len(lines) and not re.match(r"^\d+\b", lines[i]):
                qlines.append(lines[i]); i += 1
            qs.append({"qid": qid, "query": " ".join(qlines)})
        else:
            i += 1
    return qs

def main():
    # read docs
    docs_text, docs_name = read_first_existing(DOC_CANDIDATES)
    if not docs_text:
        print("No raw docs found in data/cranfield_raw/ (checked common filenames).")
        print("Put cran.all.1400 (or cran.all.1400.xml) into data/cranfield_raw/")
    # try TREC parse
    docs = parse_trec_style(docs_text) if docs_text else []
    if not docs:
        docs = parse_xml_like(docs_text) if docs_text else []
    # read queries
    qry_text, qry_name = read_first_existing(QRY_CANDIDATES)
    queries = parse_queries_text(qry_text) if qry_text else []
    # read qrels
    qrel_text, qrel_name = read_first_existing(QREL_CANDIDATES)
    qrels = parse_qrels_text(qrel_text) if qrel_text else []

    # write outputs
    with open(OUT_DIR/"corpus.jsonl", "w", encoding="utf8") as f:
        for d in docs:
            # prefer readable id (prefix if numeric)
            did = str(d.get("id"))
            f.write(json.dumps({"id": did, "title": d.get("title",""), "text": d.get("text","")}, ensure_ascii=False) + "\n")

    with open(OUT_DIR/"queries.jsonl", "w", encoding="utf8") as f:
        for q in queries:
            f.write(json.dumps({"qid": str(q.get("qid")), "query": q.get("query","")}, ensure_ascii=False) + "\n")

    with open(OUT_DIR/"qrels.jsonl", "w", encoding="utf8") as f:
        for r in qrels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Docs written: {len(docs)} (from {docs_name})")
    print(f"Queries written: {len(queries)} (from {qry_name})")
    print(f"Qrels written: {len(qrels)} (from {qrel_name})")
    print("Done.")

if __name__ == "__main__":
    main()
