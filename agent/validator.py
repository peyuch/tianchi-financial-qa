"""Validator: answer normalization, confidence check, output formatting."""
import re
from config import LOW_CONFIDENCE_THRESHOLD


def normalize_answer(raw: str, answer_format: str) -> str:
    """Normalize a raw model answer into a canonical form.

    For mcq/tf: returns the first valid A-D letter found, or empty string.
    For multi: returns sorted, deduplicated string of valid A-D letters.
    Chinese "正确"/"对" maps to A, "错误"/"错" maps to B.
    """
    letters = re.findall(r'[A-Da-d]', raw)
    if not letters:
        if "正确" in raw or "对" in raw:
            return "A"
        if "错误" in raw or "错" in raw:
            return "B"
        return ""

    upper = [l.upper() for l in letters if l.upper() in ("A", "B", "C", "D")]

    if answer_format in ("mcq", "tf"):
        return upper[0] if upper else ""
    elif answer_format == "multi":
        return "".join(sorted(set(upper)))
    else:
        return upper[0] if upper else ""


def validate_confidence(results: list[dict]) -> bool:
    """Return True if ALL results have confidence >= LOW_CONFIDENCE_THRESHOLD."""
    if not results:
        return False
    return all(
        r.get("confidence", 0) >= LOW_CONFIDENCE_THRESHOLD
        for r in results
    )


def get_low_confidence_options(results: list[dict]) -> list[str]:
    """Return list of option labels whose confidence is below threshold."""
    return [
        r["option"] for r in results
        if r.get("confidence", 0) < LOW_CONFIDENCE_THRESHOLD
    ]


def format_output_row(qid: str, answer: str,
                      prompt_tokens: int, completion_tokens: int) -> dict:
    """Format a single output row for CSV submission."""
    return {
        "qid": qid,
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_evidence_entry(qid: str, result: dict) -> dict:
    """Build an evidence-rich entry for traceability/debugging."""
    retrieval_entries = []
    for r in result.get("results", []):
        evidence = r.get("evidence", {})
        reasoning_text = (
            evidence.get("reasoning")
            or evidence.get("calculation")
            or evidence.get("years", "")
            or r.get("judgment", "")
        )
        retrieval_entries.append({
            "doc_id": evidence.get("doc_id", ""),
            "quoted_clause": evidence.get("quote", ""),
            "reasoning": str(reasoning_text),
        })

    return {
        "qid": qid,
        "answer": result.get("answer", ""),
        "evidence_retrieval": retrieval_entries,
    }
