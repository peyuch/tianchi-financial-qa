"""Retriever: Stage 1 keyword-based + Stage 2 Qwen fine-grained filtering."""
import re
import jieba
from collections import OrderedDict
from agent.qwen_client import QwenClient, build_chat_message

# Chinese stop words to filter out from keywords
_STOP_WORDS = set("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 他 她 它 们 那 些 什么 而 为 所 以 之 与 及 或 但 被 从 把 对 将 能 可以 可能 需要 已经 还 又 再 更 最 请 关于 根据 按照 下列 以下 以上 其中 包括 下列 各项 是否 目前 相关 进行 以及".split())

# Minimum character length for Chinese keywords (filter out single chars like "条", "款")
_MIN_KEYWORD_LEN = 2


def extract_keywords_from_question(question: str) -> list[str]:
    keywords = []

    # 1. Extract quoted names from 《...》
    quoted = re.findall(r'《([^》]+)》', question)
    keywords.extend(quoted)

    # 2. Use entity patterns from indexer (company names, metrics, dates)
    from agent.indexer import ENTITY_PATTERNS
    for pattern_name in ["company", "metric", "date_threshold"]:
        pattern = ENTITY_PATTERNS.get(pattern_name)
        if pattern:
            for match in pattern.findall(question):
                keywords.append(match)

    # 3. Jieba Chinese word segmentation
    # Remove punctuation for cleaner segmentation
    clean_question = re.sub(r'[，。、；：？！\s]+', ' ', question)
    words = jieba.cut(clean_question)
    for w in words:
        w = w.strip()
        if len(w) >= _MIN_KEYWORD_LEN and w not in _STOP_WORDS:
            keywords.append(w)

    # 4. Also extract 2-4 char n-grams as fallback for financial terms
    # that jieba might not know (e.g., clause numbers, product codes)
    alpha_chars = re.findall(r'[一-鿿]{2,4}', question)
    for term in alpha_chars:
        if term not in _STOP_WORDS:
            keywords.append(term)

    # Deduplicate preserving order
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
