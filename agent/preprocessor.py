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


def extract_text_pdf(filepath: str) -> str:
    """Extract text from PDF using PyMuPDF as fallback (MinerU preferred when available)."""
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
