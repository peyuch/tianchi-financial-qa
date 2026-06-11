"""Indexer: keyword inverted index + chapter tree + document summaries."""
import json
import os
import re
from pathlib import Path
from config import INDEX_DIR, SUMMARIES_DIR, DOMAIN_CONFIG

CLAUSE_PATTERN = re.compile(
    r'(第[一二三四五六七八九十百千\d]+条|第[一二三四五六七八九十百千\d]+款|'
    r'第[一二三四五六七八九十百千\d]+章|第[一二三四五六七八九十百千\d]+节)'
)

ARTICLE_NUM_PATTERN = re.compile(r'第[一二三四五六七八九十百千\d]+条')

ENTITY_PATTERNS = {
    "company": re.compile(r'(比亚迪|宁德时代|中国移动|美的集团|招商银行|中国建筑|'
                          r'平安|国寿|众安|太保|芯原|远望谷|宇信科技)'),
    "metric": re.compile(r'(营业收入|净利润|现金流[量净]额|研发投入|现金价值|'
                         r'身故保险金|退保[金金额]|免赔额|资产负债率|'
                         r'经营活动.*?现金流|归属于.*?净利润)'),
    "date_threshold": re.compile(r'(施行之日|生效之日|发布之日|自.*?起|'
                                 r'20\d{2}年|资产负债率超过\s*[0-9.]+%|'
                                 r'[0-9.]+亿[元]|[0-9.]+万[元])'),
}


def build_keyword_index(doc_id: str, text: str, domain: str | None = None,
                        pdf_backend: str = "pymupdf") -> dict:
    """Build keyword index entries for a single document.

    PyMuPDF4LLM and MinerU both output markdown → simple \n\n split.
    PyMuPDF legacy raw text uses domain-aware smart splitting.
    """
    if pdf_backend == "pymupdf-legacy" and domain:
        from agent.preprocessor import split_paragraphs_pymupdf
        paragraphs = split_paragraphs_pymupdf(text, domain)
    else:
        # Markdown from PyMuPDF4LLM or MinerU — \n\n split works
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    entries = []
    for i, para in enumerate(paragraphs):
        if len(para) < 10:
            continue
        entries.append({"para_id": i, "text": para})
    return {doc_id: entries}


def extract_headings(text: str) -> list[dict]:
    headings = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            headings.append({"level": level, "text": heading_text})
    return headings


def extract_summary(doc_id: str, domain: str, text: str) -> dict:
    headings = extract_headings(text)
    heading_text = " ".join(h["text"] for h in headings)
    first_section = text[:500]

    keywords = set()
    for pattern in ENTITY_PATTERNS.values():
        for match in pattern.findall(heading_text + first_section):
            keywords.add(match)

    for match in ARTICLE_NUM_PATTERN.findall(first_section):
        keywords.add(match)

    toc = [h["text"] for h in headings[:20]]

    year_match = re.search(r'20(\d{2})', heading_text)
    year = f"20{year_match.group(1)}" if year_match else None

    company = None
    company_match = ENTITY_PATTERNS["company"].search(heading_text)
    if company_match:
        company = company_match.group(0)

    return {
        "doc_id": doc_id,
        "domain": domain,
        "summary_keywords": sorted(keywords)[:20],
        "toc": toc,
        "meta": {
            "year": year,
            "company": company,
        },
    }


def index_document(doc_id: str, domain: str, md_path: str) -> dict:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    return build_keyword_index(doc_id, text)


def search_keyword_index(index: dict, query: str) -> list[dict]:
    results = []
    query_lower = query.lower()
    for doc_id, entries in index.items():
        for entry in entries:
            if query_lower in entry["text"].lower():
                results.append({
                    "doc_id": doc_id,
                    "para_id": entry["para_id"],
                    "text": entry["text"],
                })
    return results


def search_multi_keyword(index: dict, keywords: list[str]) -> list[dict]:
    seen = set()
    results = []
    for kw in keywords:
        for hit in search_keyword_index(index, kw):
            key = (hit["doc_id"], hit["para_id"])
            if key not in seen:
                seen.add(key)
                results.append(hit)
    return results


def build_all_indexes(doc_md_map: dict[str, tuple[str, str]]) -> tuple[dict, list[dict]]:
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    keyword_index = {}
    summaries = []

    for doc_id, (domain, md_path) in doc_md_map.items():
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_index = build_keyword_index(doc_id, text)
        keyword_index.update(doc_index)

        summary = extract_summary(doc_id, domain, text)
        summaries.append(summary)

    with open(os.path.join(INDEX_DIR, "keyword_index.json"), "w", encoding="utf-8") as f:
        json.dump(keyword_index, f, ensure_ascii=False)

    with open(os.path.join(SUMMARIES_DIR, "doc_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return keyword_index, summaries
