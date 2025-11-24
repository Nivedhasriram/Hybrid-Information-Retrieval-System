# scripts/convert_queries_xml.py
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import re

RAW_QRY = Path("data/cranfield_raw/cran.qry")   # your actual file
OUT = Path("data/cranfield/queries.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def parse_queries(path):
    tree = ET.parse(path)
    root = tree.getroot()

    queries = []
    # find all <top> blocks
    for top in root.findall("top"):
        num = top.find("num")
        title = top.find("title")

        if num is None or title is None:
            continue

        # Extract numeric ID
        num_text = "".join(re.findall(r"\d+", num.text))
        qid = f"Q{num_text}"

        # Clean title text
        qtext = " ".join(title.text.split())

        queries.append({"qid": qid, "query": qtext})

    return queries


def main():
    if not RAW_QRY.exists():
        print("ERROR: Query file not found:", RAW_QRY)
        return

    queries = parse_queries(RAW_QRY)
    print(f"Parsed {len(queries)} queries")

    with OUT.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
