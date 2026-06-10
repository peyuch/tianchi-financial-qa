"""Retriever: Stage 1 keyword-based + Stage 2 Qwen fine-grained filtering."""
import re
from collections import OrderedDict
from agent.qwen_client import QwenClient, build_chat_message


def extract_keywords_from_question(question: str) -> list[str]:
    keywords = []
    quoted = re.findall(r'《([^》]+)》', question)
    keywords.extend(quoted)
    from agent.indexer import ENTITY_PATTERNS
    for pattern_name in ["company", "metric", "date_threshold"]:
        pattern = ENTITY_PATTERNS.get(pattern_name)
        if pattern:
            for match in pattern.findall(question):
                keywords.append(match)
    chinese_terms = re.findall(r'[一-鿿]{3,}', question)
    keywords.extend(chinese_terms)
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def truncate_paragraphs(paragraphs: list[dict], max_tokens: int = 500) -> list[dict]:
    result = []
    for p in paragraphs:
        text = p["text"]
        if len(text) > max_tokens:
            text = text[:max_tokens] + "..."
        result.append({**p, "text": text})
    return result


def stage1_retrieve(index: dict, doc_ids: list[str], question: str) -> list[dict]:
    keywords = extract_keywords_from_question(question)
    results = []
    seen = set()
    for kw in keywords:
        for doc_id in doc_ids:
            if doc_id not in index:
                continue
            for entry in index[doc_id]:
                if kw.lower() in entry["text"].lower():
                    key = (doc_id, entry["para_id"])
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "doc_id": doc_id,
                            "para_id": entry["para_id"],
                            "text": entry["text"],
                        })
        if len(results) >= 30:
            break
    return truncate_paragraphs(results[:30])


def allocate_per_doc(evidence: list[dict], per_doc_cap: int,
                     max_docs: int) -> list[dict]:
    doc_groups = OrderedDict()
    for e in evidence:
        doc_id = e["doc_id"]
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append(e)

    result = []
    for doc_id, entries in list(doc_groups.items())[:max_docs]:
        doc_chars = 0
        for entry in entries:
            text = entry["text"]
            if doc_chars + len(text) > per_doc_cap:
                remaining = per_doc_cap - doc_chars
                if remaining > 50:
                    entry = {**entry, "text": text[:remaining] + "..."}
                    result.append(entry)
                break
            result.append(entry)
            doc_chars += len(text)
    return result


def stage2_filter(client: QwenClient, candidates: list[dict],
                  question: str, options: dict) -> tuple[list[dict], dict]:
    if len(candidates) <= 5:
        return candidates, {"input_tokens": 0, "output_tokens": 0}

    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "stage2_filter.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    options_text = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    candidates_text = "\n\n".join(
        f"#{i}: [{c['doc_id']}] {c['text']}"
        for i, c in enumerate(candidates)
    )

    prompt = template.format(
        question=question,
        options=options_text,
        candidates=candidates_text,
    )

    response = client.chat(
        messages=[build_chat_message("user", prompt)],
        temperature=0.1,
        max_tokens=500,
    )

    content = response["content"].strip()
    numbers = re.findall(r'\d+', content)
    ranked_ids = [int(n) for n in numbers if int(n) < len(candidates)]

    top5 = []
    seen = set()
    for idx in ranked_ids[:5]:
        if idx not in seen:
            seen.add(idx)
            top5.append(candidates[idx])

    token_usage = {
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
    }
    return top5, token_usage


def should_skip_stage2(candidates: list[dict], question: str) -> bool:
    if len(candidates) <= 5:
        return True
    from agent.indexer import ARTICLE_NUM_PATTERN
    clause_nums = ARTICLE_NUM_PATTERN.findall(question)
    if clause_nums:
        top3_texts = " ".join(c["text"] for c in candidates[:3])
        for cn in clause_nums:
            if cn in top3_texts:
                return True
    return False
