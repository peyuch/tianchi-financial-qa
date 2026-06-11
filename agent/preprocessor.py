"""Preprocessor: PDF/HTML/TXT → unified markdown."""
import os
import re
from pathlib import Path
from config import RAW_DIR, PROCESSED_DIR, DOC_EXTENSION_MAP

DOMAIN_DIRS = {
    "insurance": os.path.join(RAW_DIR, "insurance"),
    "financial_reports": os.path.join(RAW_DIR, "financial_reports"),
    "financial_contracts": os.path.join(RAW_DIR, "financial_contracts"),
    "research": os.path.join(RAW_DIR, "research"),
    "regulatory": os.path.join(RAW_DIR, "regulatory"),
}

REGULATORY_SUBDIRS = ["txt", "html", "attachments"]


def resolve_doc_path(domain: str, doc_id: str) -> str | None:
    """Find the actual file path for a given domain and doc_id."""
    extensions = DOC_EXTENSION_MAP.get(domain, [".pdf"])
    base_dir = DOMAIN_DIRS[domain]

    # For regulatory, search subdirectories
    if domain == "regulatory":
        for subdir in REGULATORY_SUBDIRS:
            search_dir = os.path.join(base_dir, subdir)
            if not os.path.isdir(search_dir):
                continue
            for ext in extensions:
                path = os.path.join(search_dir, f"{doc_id}{ext}")
                if os.path.isfile(path):
                    return path
        # Also search the main directory itself
        for ext in extensions:
            path = os.path.join(base_dir, f"{doc_id}{ext}")
            if os.path.isfile(path):
                return path
        return None

    # For other domains: simple lookup in domain directory
    for ext in extensions:
        path = os.path.join(base_dir, f"{doc_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def _extract_pdf_mineru(filepath: str) -> str:
    """MinerU CLI — structured markdown with full table preservation."""
    import subprocess, tempfile, shutil
    output_dir = tempfile.mkdtemp(prefix="mineru_")
    try:
        result = subprocess.run(
            ["mineru", "-p", filepath, "-o", output_dir, "-b", "pipeline"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MinerU failed: {result.stderr[:500]}")
        basename = os.path.splitext(os.path.basename(filepath))[0]
        for alt in [
            os.path.join(output_dir, basename, basename, f"{basename}.md"),
            os.path.join(output_dir, basename, f"{basename}.md"),
        ]:
            if os.path.isfile(alt):
                with open(alt, "r", encoding="utf-8") as f:
                    return f.read()
        raise RuntimeError("MinerU output not found")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _extract_pdf_pymupdf4llm(filepath: str) -> str:
    """PyMuPDF4LLM — markdown with auto-detected tables, headings, paragraphs."""
    import pymupdf4llm
    return pymupdf4llm.to_markdown(filepath)


def _extract_pdf_fitz_raw(filepath: str) -> str:
    """Legacy PyMuPDF raw text — only as last-resort fallback."""
    import fitz
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def extract_text_pdf(filepath: str) -> str:
    """Extract PDF as markdown.

    Backend (env PDF_BACKEND):
    - "pymupdf4llm" → PyMuPDF4LLM (markdown, tables, fast, DEFAULT)
    - "mineru"      → MinerU CLI (best structure, GPU, slow)
    - "pymupdf"     → PyMuPDF fitz raw text (no structure, fastest)
    """
    import os as _os
    backend = _os.environ.get("PDF_BACKEND", "pymupdf4llm")

    if backend == "mineru":
        import shutil
        if shutil.which("mineru"):
            try:
                return _extract_pdf_mineru(filepath)
            except Exception as e:
                print(f"  MinerU failed ({e}), falling back to PyMuPDF4LLM")
    elif backend == "pymupdf":
        return _extract_pdf_fitz_raw(filepath)

    # Default: PyMuPDF4LLM
    return _extract_pdf_pymupdf4llm(filepath)


def extract_text_html(filepath: str) -> str:
    """Extract plain text from HTML, preserving heading structure."""
    from bs4 import BeautifulSoup
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"]):
        tag = el.name
        text = el.get_text(strip=True)
        if not text:
            continue
        if tag.startswith("h"):
            level = int(tag[1])
            prefix = "#" * level
            lines.append(f"\n{prefix} {text}\n")
        else:
            lines.append(text)

    return "\n\n".join(lines)


def extract_text_txt(filepath: str) -> str:
    """Read plain text file directly."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_paragraphs_pymupdf(text: str, domain: str) -> list[str]:
    """Smart paragraph splitting for PyMuPDF raw text output.

    PyMuPDF dumps pages as flat text without structure. This reconstructs:
    - Table rows (metric name + value lines glued together)
    - Multi-column table blocks (space-aligned PyMuPDF output)
    - Natural paragraph boundaries
    - Domain-specific max chunk sizes to avoid oversized paragraphs
    """
    import re

    # ── Step 1: Clean page headers/footers ──
    text = re.sub(
        r'[一-鿿]{2,20}(股份有限公司|集团|保险|证券)[^\n]{0,30}第\s*\d+\s*页\n?',
        '', text
    )
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)

    # ── Step 2: Reconstruct table structure ──
    lines = text.split('\n')
    paragraphs = []
    i = 0

    # Metric keywords for table row detection
    metric_pattern = re.compile(
        r'营业收入|净利润|总资产|净资产|现金流|研发|每股|保费|赔付|'
        r'发行金额|票面利率|资产负债率|毛利率|营业成本|归属于|'
        r'现金分红|利润分配|募集资金|发行规模|信用评级|转股价格|'
        r'赎回条款|回售条款|股票代码|发行日期|身故保险金|现金价值|'
        r'基本保额|账户价值|免赔额'
    )
    number_pattern = re.compile(r'^[\d,.\s\-（）%亿万元千百]+$')

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Detect table: metric name + next line is a number
        is_metric = bool(metric_pattern.search(line))
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        is_number = bool(number_pattern.match(next_line)) if next_line else False

        if is_metric and is_number:
            table_block = [line + ' ' + next_line]
            i += 2
            while i < len(lines):
                curr = lines[i].strip()
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
                if curr and bool(number_pattern.match(nxt)):
                    table_block.append(curr + ' ' + nxt)
                    i += 2
                else:
                    break
            paragraphs.append('\n'.join(table_block))

        # Detect space-aligned multi-column table rows (PyMuPDF column format)
        elif re.match(r'^.{2,15}\s{3,}.{1,20}\s{3,}', line):
            table_lines = [line]
            i += 1
            while i < len(lines):
                next_l = lines[i].strip()
                if next_l and re.match(r'^.{2,15}\s{3,}', next_l):
                    table_lines.append(next_l)
                    i += 1
                else:
                    break
            paragraphs.append('\n'.join(table_lines))

        else:
            # Normal text: accumulate until blank line or next metric row
            text_block = [line]
            i += 1
            while i < len(lines):
                next_l = lines[i].strip()
                if not next_l:
                    i += 1
                    break
                if metric_pattern.search(next_l):
                    break
                text_block.append(next_l)
                i += 1
            block = ' '.join(text_block)
            if len(block) > 15:
                paragraphs.append(block)

    # ── Step 3: Domain-aware max chunk size ──
    max_chars = {
        'financial_reports': 1500,
        'financial_contracts': 1200,
        'research': 1000,
        'insurance': 800,
        'regulatory': 600,
    }.get(domain, 1000)

    result = []
    for p in paragraphs:
        if len(p) <= max_chars:
            result.append(p)
        else:
            sentences = re.split(r'(?<=[。；\n])', p)
            chunk, chunk_len = [], 0
            for s in sentences:
                if chunk_len + len(s) > max_chars and chunk:
                    result.append(''.join(chunk))
                    chunk, chunk_len = [s], len(s)
                else:
                    chunk.append(s)
                    chunk_len += len(s)
            if chunk:
                result.append(''.join(chunk))

    return [p for p in result if len(p.strip()) > 15]


def merge_orphan_number_lines(md_content: str) -> str:
    """Merge orphan number lines into previous label lines.

    Fixes the table fragmentation problem where PyMuPDF/MinerU splits
    "营业收入（千元）" and "362,012,554" into separate lines.
    After merging: "营业收入（千元） 362,012,554"
    """
    lines = md_content.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_next = lines[i + 1].strip() if i + 1 < len(lines) else ""
        # If next line is an orphan number (digits/commas/spaces/dots/dashes only)
        # and current line has meaningful content (>2 chars), merge them
        if (i + 1 < len(lines)
                and re.match(r'^[\d,，.\s\-%％]+$', stripped_next)
                and stripped_next
                and len(line.strip()) > 2):
            merged.append(line.rstrip() + ' ' + stripped_next)
            i += 2
        else:
            merged.append(line)
            i += 1
    return '\n'.join(merged)


EXTRACTORS = {
    ".pdf": extract_text_pdf,
    ".PDF": extract_text_pdf,
    ".html": extract_text_html,
    ".txt": extract_text_txt,
}


def preprocess_document(filepath: str, file_type: str | None = None) -> str:
    """Extract text from a document and return path to processed .md file."""
    if file_type is None:
        _, ext = os.path.splitext(filepath)
        file_type = ext

    # Normalize: ensure leading dot (e.g. "txt" → ".txt")
    if not file_type.startswith("."):
        file_type = f".{file_type}"

    extractor = EXTRACTORS.get(file_type)
    if extractor is None:
        raise ValueError(f"No extractor for file type: {file_type}")

    text = extractor(filepath)
    # Only run orphan merge for legacy raw-text backend (pymupdf)
    if os.environ.get("PDF_BACKEND", "pymupdf4llm") == "pymupdf":
        text = merge_orphan_number_lines(text)

    doc_id = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


def preprocess_all_docs(doc_list: list[tuple[str, str]]) -> dict[str, str]:
    """Preprocess a list of (domain, doc_id) tuples.
    Returns dict mapping doc_id → processed .md path.
    """
    results = {}
    for domain, doc_id in doc_list:
        path = resolve_doc_path(domain, doc_id)
        if path is None:
            print(f"WARNING: doc not found: domain={domain} doc_id={doc_id}")
            continue
        ext = os.path.splitext(path)[1]
        try:
            md_path = preprocess_document(path, ext)
            results[doc_id] = md_path
            print(f"  OK: {doc_id} → {md_path}")
        except Exception as e:
            print(f"  FAIL: {doc_id} ({path}): {e}")
    return results
