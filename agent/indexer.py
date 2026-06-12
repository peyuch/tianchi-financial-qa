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


def _strip_markdown_noise(text: str) -> str:
    """Replace markdown symbols with spaces for cleaner keyword matching."""
    import re
    return re.sub(r'[|#*>\-]+', ' ', text)


def preprocess_and_chunk_md(text: str, doc_id: str) -> list[dict]:
    """Multi-domain adaptive chunking with table semantic flattening.

    - Table rows: sub-chunked every 8 rows, headers injected into each chunk
    - Cross-page header inheritance
    - Year+metric compound search terms (e.g. "2025年营业收入")
    - Dual-track output: text (markdown + path) / search_text (cleaned, dense terms)
    """
    import re

    # ── STEP 1: Global noise cleaning ──
    text = re.sub(
        r'\*\*-----\s*Start of picture text\s*-----\*\*(?:<br>)?[\s\S]*?'
        r'\*\*-----\s*End of picture text\s*-----\*\*(?:<br>)?', '', text
    )
    text = re.sub(r'\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*', '', text)
    text = re.sub(r'[•●─]', '', text)
    text = re.sub(r'-\s*\[(.*?)\]', r'- \1', text)

    # ── STEP 2: State-machine chunking ──
    lines = text.split('\n')
    chunks = []
    current_section = ""
    current_clause = ""
    current_statute = ""
    table_buffer = []
    last_known_table_header = None
    in_table = False
    text_buffer = []

    def _emit_text():
        nonlocal text_buffer
        if not text_buffer:
            return
        raw_md = '\n'.join(text_buffer).strip()
        text_buffer = []
        if len(raw_md) < 5:
            return
        path_prefix = f"【文档ID: {doc_id} | 路径: {current_section} -> {current_clause or current_statute or '正文'}】\n"
        search_text = raw_md.replace('<br>', ' ').replace('<br />', ' ')
        search_text = re.sub(r'[^a-zA-Z0-9一-龥\s]', ' ', search_text)
        chunks.append({
            "text": path_prefix + raw_md,
            "search_text": " ".join(search_text.split()),
            "path": current_section,
        })

    def _emit_table():
        nonlocal table_buffer, last_known_table_header
        t_lines = list(table_buffer)
        table_buffer = []
        if not t_lines:
            return

        has_header = len(t_lines) > 1 and '---' in t_lines[1]
        if has_header:
            header_lines = t_lines[:2]
            last_known_table_header = header_lines
            data_lines = t_lines[2:]
        elif last_known_table_header:
            header_lines = last_known_table_header
            data_lines = t_lines
        else:
            header_lines = []
            data_lines = t_lines

        if not data_lines:
            return

        # Parse column headers (e.g. ['指标', '2025年', '2024年'])
        col_headers = []
        if header_lines:
            col_headers = [c.strip() for c in header_lines[0].split('|')[1:-1]]

        row_group_size = 8
        for i in range(0, len(data_lines), row_group_size):
            sub_data = data_lines[i:i + row_group_size]
            full_table_md = '\n'.join(header_lines + sub_data) if header_lines else '\n'.join(sub_data)
            path_prefix = f"【文档ID: {doc_id} | 路径: {current_section} -> 表格数据】\n"

            # Semantic flattening: cross-bind year headers with metric names
            search_terms = []
            for row in sub_data:
                row_cols = [c.strip() for c in row.split('|')[1:-1]]
                if not row_cols:
                    continue
                row_title = row_cols[0]
                search_terms.append(row_title)
                for ci, cv in enumerate(row_cols[1:]):
                    hi = ci + 1
                    if hi < len(col_headers):
                        search_terms.append(f"{col_headers[hi]}{row_title}")
                    search_terms.append(cv)

            flat_search = " ".join(search_terms)
            flat_search = re.sub(r'[^a-zA-Z0-9一-龥\s]', ' ', flat_search)

            chunks.append({
                "text": path_prefix + full_table_md,
                "search_text": " ".join(flat_search.split()),
                "path": current_section,
            })

    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith('|') and stripped.endswith('|')

        if is_table_row:
            if not in_table:
                _emit_text()
                in_table = True
            table_buffer.append(line)
            continue
        else:
            if in_table:
                _emit_table()
                in_table = False

        if not stripped:
            continue

        header_match = re.match(r'^(#{1,3})\s+(.*)', stripped)
        ins_match = re.match(r'^-\s*(\d+\.\d+)\s+(.*)', stripped)
        reg_match = re.match(r'^第([一二三四五六七八九十百千\d]+)条\s*(.*)', stripped)

        is_trigger = header_match or ins_match or reg_match
        if is_trigger:
            _emit_text()
            text_buffer = [line]

        if header_match:
            current_section = header_match.group(2).strip()
            current_clause, current_statute = "", ""
        elif ins_match:
            current_clause = f"{ins_match.group(1)} {ins_match.group(2)}"
            current_statute = ""
        elif reg_match:
            current_statute = f"第{reg_match.group(1)}条 {reg_match.group(2)[:15]}"
            current_clause = ""
        elif not is_trigger:
            text_buffer.append(line)

    if in_table:
        _emit_table()
    else:
        _emit_text()

    return chunks


def build_keyword_index(doc_id: str, text: str, domain: str | None = None,
                        pdf_backend: str = "pymupdf") -> dict:
    """Build keyword index with dual-track entries.

    pymupdf4llm/mineru: heading-aware chunking, ancestor path injection, dual-track.
    pymupdf (legacy): domain-aware raw text splitting.
    """
    if pdf_backend == "pymupdf" and domain:
        from agent.preprocessor import split_paragraphs_pymupdf
        paragraphs_raw = split_paragraphs_pymupdf(text, domain)
        paragraphs = [{"text": p, "search_text": p, "path": ""} for p in paragraphs_raw]
    else:
        paragraphs = preprocess_and_chunk_md(text, doc_id)

    entries = []
    for i, para in enumerate(paragraphs):
        search = para.get("search_text", para["text"])
        if len(search.strip()) < 10:
            continue
        entries.append({
            "para_id": i,
            "text": para["text"],
            "search_text": para.get("search_text", para["text"]),
        })
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
