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


def extract_text_pdf_mineru(filepath: str) -> str:
    """Extract text from PDF using MinerU (GPU-accelerated on CUDA, auto-fallback to CPU).

    MinerU CLI auto-detects GPU. On GPU server, no -b flag is needed.
    Output is cached in data/processed/ so subsequent runs skip parsing.
    """
    import subprocess, tempfile, shutil

    output_dir = tempfile.mkdtemp(prefix="mineru_")
    try:
        result = subprocess.run(
            ["mineru", "-p", filepath, "-o", output_dir, "-b", "pipeline"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MinerU failed: {result.stderr[:500]}")

        # MinerU output: output_dir/{basename}/{basename}.md
        basename = os.path.splitext(os.path.basename(filepath))[0]
        md_path = os.path.join(output_dir, basename, basename, f"{basename}.md")
        if not os.path.isfile(md_path):
            # Try alternative path pattern for some MinerU versions
            alt = os.path.join(output_dir, basename, f"{basename}.md")
            if os.path.isfile(alt):
                md_path = alt
            else:
                raise RuntimeError(f"MinerU output not found at {md_path} or {alt}")

        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def extract_text_pdf(filepath: str) -> str:
    """Extract text from PDF. Backend controlled by PDF_BACKEND env var."""
    import os as _os
    backend = _os.environ.get("PDF_BACKEND", "mineru")

    if backend == "mineru":
        try:
            import shutil
            if shutil.which("mineru"):
                return extract_text_pdf_mineru(filepath)
        except Exception as e:
            print(f"  MinerU failed ({e}), falling back to PyMuPDF")

    # PyMuPDF (either selected or fallback)
    import fitz
    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


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
