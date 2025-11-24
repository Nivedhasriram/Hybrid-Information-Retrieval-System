#!/usr/bin/env python3
import sys
import os
import subprocess
import json
import importlib
import traceback
import ast
from pathlib import Path
from datetime import datetime

# --- make imports robust: ensure project root is on sys.path ---
from pathlib import Path
import sys, os
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ----------------------------------------------------------------

ROOT = Path(".").resolve()
TOOLS_DIR = ROOT / "tools"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)
REPORT_PATH = REPORT_DIR / "check_report.txt"

# Path to raw cran file you uploaded (keeps the original absolute path)
RAW_CRAN_PATH = "/mnt/data/cran.all.1400.xml"

# helper to log
log_lines = []
def log(s=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    log_lines.append(line)

def run_cmd(cmd, cwd=None, env=None, timeout=600):
    """Run shell command, return (rc, stdout, stderr)"""
    try:
        proc = subprocess.run(cmd, cwd=cwd or ROOT, env=env, shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return 1, "", f"Exception running command: {e}\n{traceback.format_exc()}"

def safe_import(module_name):
    try:
        m = importlib.import_module(module_name)
        return m, None
    except Exception as e:
        return None, traceback.format_exc()

def write_report():
    REPORT_PATH.write_text("\n".join(log_lines), encoding="utf8")
    log(f"Full report written to: {REPORT_PATH}")

def main():
    log("START PROJECT CHECK")
    log(f"Project root: {ROOT}")
    log(f"Raw cran file (expected): {RAW_CRAN_PATH}")
    log("")

    # 1) Basic environment
    log("1) Python & venv")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version.replace(os.linesep,' ')}")
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    log(f"Virtualenv active: {bool(venv)} ({venv})")
    log("")

    # 2) Basic filesystem checks
    log("2) Files / directories sanity check")
    want = [
        "src", "src/data_loader.py", "src/indexer.py", "src/semantic_index.py",
        "src/hybrid_retrieve.py", "src/app.py", "scripts/convert_cranfield.py",
        "scripts/convert_queries_xml.py", "data/cranfield", "data/cranfield/corpus.jsonl"
    ]
    for p in want:
        exists = (ROOT / p).exists()
        log(f"{p:40}  -> {'OK' if exists else 'MISSING'}")
    # Raw cran file path
    log(f"raw cran path exists: {Path(RAW_CRAN_PATH).exists()}")
    log("")

    # 3) Run XML -> JSONL converters if present (safe: only if file is present)
    log("3) Run conversion scripts (if present)")
    # try convert_queries_xml.py first
    c_qry = ROOT / "scripts" / "convert_queries_xml.py"
    if c_qry.exists():
        log("Running scripts/convert_queries_xml.py ...")
        rc, out, err = run_cmd(f"python3 {c_qry}", cwd=ROOT)
        log(f"rc={rc}")
        if out: log("OUT: " + out.splitlines()[0])
        if err: log("ERR: " + (err.splitlines()[0] if err else ""))
    else:
        log("scripts/convert_queries_xml.py not found, skipping")

    # try a generic convert_cranfield script or the module converted earlier
    conv = ROOT / "scripts" / "convert_cranfield.py"
    if conv.exists():
        log("Running scripts/convert_cranfield.py ...")
        rc, out, err = run_cmd(f"python3 {conv}", cwd=ROOT)
        log(f"rc={rc}")
        if out: log("OUT: " + (out.splitlines()[0] if out else ""))
        if err: log("ERR: " + (err.splitlines()[0] if err else ""))
    else:
        # try module style
        rc, out, err = run_cmd("python3 -m scripts.convert_cranfield", cwd=ROOT)
        log(f"convert_cranfield module rc={rc}")
        if out: log("OUT: " + (out.splitlines()[0] if out else ""))
        if err and "No module named" not in err:
            log("ERR: " + (err.splitlines()[0] if err else ""))
    log("")

    # 4) Attempt to import data_loader and load cranfield
    log("4) Import src.data_loader and load data")
    m, err = safe_import("src.data_loader")
    if not m:
        log("FAILED to import src.data_loader:")
        log(err)
        write_report()
        return
    try:
        load_fn = getattr(m, "load_cranfield", None)
        if not load_fn:
            log("src.data_loader has no function load_cranfield()")
        else:
            log("Calling load_cranfield() ...")
            corpus_df, queries_df, qrels_map = load_fn()
            log(f"Loaded corpus rows: {len(corpus_df)}")
            log(f"Loaded queries rows: {len(queries_df)}")
            log(f"Loaded qrels entries: {len(qrels_map)}")
            if len(corpus_df) > 0:
                log(f"Sample doc id/title: {corpus_df.iloc[0]['id']} / {str(corpus_df.iloc[0].get('title',''))[:80]}")
            if len(queries_df) > 0:
                log(f"Sample query: {queries_df.iloc[0].get('qid')} / {queries_df.iloc[0].get('query')[:90]}")
    except Exception as e:
        log("Exception while running load_cranfield():")
        log(traceback.format_exc())
    log("")

    # 5) Try to build or load TF-IDF index via src.indexer
    log("5) TF-IDF index build/load check (src.indexer)")
    midx, ierr = safe_import("src.indexer")
    if not midx:
        log("FAILED to import src.indexer")
        log(ierr or "")
    else:
        # attempt to call load_index (preferred) or build_index
        try:
            if hasattr(midx, "load_index"):
                log("Calling src.indexer.load_index() ...")
                vec, X, docs = midx.load_index()
                shape = getattr(X, "shape", None)
                log(f"Loaded TF-IDF: matrix shape {shape}, docs rows {len(docs)}")
            else:
                log("Calling src.indexer.build_index() ...")
                vec, X, docs = midx.build_index()
                log(f"Built TF-IDF: matrix shape {getattr(X,'shape',None)}")
        except Exception:
            log("Exception in indexer:")
            log(traceback.format_exc())
    log("")

    # 6) Try to build/load semantic index
    log("6) Semantic index check (src.semantic_index)")
    msem, serr = safe_import("src.semantic_index")
    if not msem:
        log("FAILED to import src.semantic_index")
        log(serr or "")
    else:
        try:
            # many implementations provide a build_semantic_index() or similar
            if hasattr(msem, "load_semantic_index"):
                log("Calling src.semantic_index.load_semantic_index() ...")
                emb, docs2, model = msem.load_semantic_index()
                log(f"Loaded embeddings shape: {getattr(emb,'shape',None)}")
            elif hasattr(msem, "build_semantic_index"):
                log("Calling src.semantic_index.build_semantic_index() ...")
                emb, docs2, model = msem.build_semantic_index()
                log(f"Built embeddings shape: {getattr(emb,'shape',None)}")
            else:
                log("No load/build function found in src.semantic_index")
        except Exception:
            log("Exception in semantic_index:")
            log(traceback.format_exc())
    log("")

    # 7) Run a hybrid_search smoke test if hybrid_retrieve available
    log("7) Hybrid retrieval smoke test (src.hybrid_retrieve.hybrid_search)")
    mh, herr = safe_import("src.hybrid_retrieve")
    if not mh:
        log("FAILED to import src.hybrid_retrieve")
        log(herr or "")
    else:
        try:
            hs = getattr(mh, "hybrid_search", None)
            if not hs:
                log("No hybrid_search() in src.hybrid_retrieve")
            else:
                sample_q = None
                # choose a query from queries_df if present
                try:
                    if 'queries_df' in locals() and len(queries_df) > 0:
                        sample_q = queries_df.iloc[0]['query']
                    else:
                        sample_q = "airplane engine"
                    log(f"Running hybrid_search on query: {sample_q[:120]}")
                    res = hs(sample_q, topk=5)
                    # res may be a DataFrame or list
                    try:
                        rlen = len(res)
                    except Exception:
                        rlen = "unknown"
                    log(f"hybrid_search returned rows: {rlen}")
                    if hasattr(res, "head"):
                        log("Sample results (id,title):")
                        lines = []
                        for r in res.head(5).to_dict(orient="records"):
                            lines.append(f"  {r.get('id')}: {str(r.get('title',''))[:80]}")
                        for ln in lines:
                            log(ln)
                except Exception:
                    log("Exception while running hybrid_search:")
                    log(traceback.format_exc())
        except Exception:
            log("Exception in hybrid_retrieve module import/runtime:")
            log(traceback.format_exc())
    log("")

    # 8) Quick eval: compute P@5 and nDCG@5 for first few queries if qrels present
    log("8) Quick evaluation if qrels and results available")
    try:
        if 'qrels_map' in locals() and qrels_map and 'mh' in locals() and hasattr(mh, "hybrid_search"):
            from math import log as _log
            def dcg(scores):
                return sum((2**s - 1) / _log(i+2, 2) for i,s in enumerate(scores))
            def ndcg_at_k(rel_list, k=5):
                actual = rel_list[:k]
                ideal = sorted(rel_list, reverse=True)[:k]
                if sum(ideal) == 0:
                    return 0.0
                return dcg(actual)/dcg(ideal)

            eval_qids = list(qrels_map.keys())[:5]
            for qid in eval_qids:
                qrow = None
                if 'queries_df' in locals():
                    qrows = queries_df[queries_df['qid']==qid]
                    if len(qrows)>0:
                        qrow = qrows.iloc[0]['query']
                if not qrow:
                    continue
                res = mh.hybrid_search(qrow, topk=10)
                # extract returned doc ids (assume column 'id' or 'docid' or index)
                returned_ids = []
                try:
                    if hasattr(res, "to_dict"):
                        returned_ids = [str(r['id']) for r in res.to_dict(orient="records")]
                    elif isinstance(res, list):
                        returned_ids = [str(x.get('id') or x.get('docid') or x) for x in res]
                except Exception:
                    returned_ids = []
                # build relevance list (1 if returned id in qrels_map[qid])
                rels = [1 if rid in qrels_map.get(qid, []) else 0 for rid in returned_ids]
                p5 = sum(rels[:5])/5.0 if len(rels)>=5 else sum(rels)/5.0
                ndcg5 = ndcg_at_k(rels, k=5)
                log(f"Eval Q {qid}: P@5={p5:.3f}, nDCG@5={ndcg5:.3f}")
        else:
            log("Skipping eval: qrels or hybrid_search not available")
    except Exception:
        log("Exception during quick eval:")
        log(traceback.format_exc())
    log("")

    # 9) Syntax check of src/app.py (parse only, don't execute)
    log("9) Syntax check for src/app.py")
    app_file = ROOT / "src" / "app.py"
    if app_file.exists():
        try:
            ast.parse(app_file.read_text(encoding="utf8"))
            log("src/app.py: syntax OK (parsed)")
        except Exception:
            log("src/app.py: syntax error / parse failed")
            log(traceback.format_exc())
    else:
        log("src/app.py does not exist")
    log("")

    # 10) Try to import src.app safely (optional) - do not execute top-level heavy code
    log("10) Optional: try importing src.app (may run Streamlit code) - SKIPPED by default")
    log("If you want to attempt import that runs side-effects, re-run the script with IMPORT_APP=1 env var.")
    if os.environ.get("IMPORT_APP") == "1":
        log("IMPORT_APP=1 set; attempting import src.app (catching exceptions)...")
        try:
            import importlib
            importlib.reload(importlib.import_module("src.app"))
            log("Imported src.app (no exception)")
        except Exception:
            log("Exception importing src.app:")
            log(traceback.format_exc())
    else:
        log("Skipping import of src.app (to avoid server start).")
    log("")

    # done
    log("PROJECT CHECK COMPLETE")
    write_report()

if __name__ == "__main__":
    main()
