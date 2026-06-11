"""Reasoner: batch CoT reasoning with domain-specific prompts."""
import json
import os
import re
from agent.qwen_client import QwenClient, build_chat_message


def build_reasoning_prompt(prompt_name: str, evidence: list[dict],
                           question: str, options: dict) -> str:
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", f"{prompt_name}.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    evidence_text = "\n\n---\n\n".join(
        f"[{e['doc_id']}]\n{e['text']}" for e in evidence
    )
    options_text = "\n".join(
        f"{k}. {v}" for k, v in sorted(options.items())
    )

    return template.format(
        evidence=evidence_text,
        question=question,
        options=options_text,
    )


def parse_reasoning_response(content: str, answer_format: str) -> dict:
    data = None
    json_match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', content)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    if data is None:
        try:
            data = json.loads(content.strip())
        except json.JSONDecodeError:
            return parse_reasoning_fallback(content, answer_format)

    results = data.get("results", [])

    # ── Always derive answer from individual option judgments ──
    # Never trust the LLM's "answer" field — option judgments and
    # aggregated answer often disagree (Option Judge ≠ Aggregator).
    answer = _derive_answer_from_judgments(results, answer_format)

    return {
        "answer": answer,
        "results": results,
        "raw": content,
    }


def _derive_answer_from_judgments(results: list[dict], answer_format: str) -> str:
    """Programmatically derive final answer from per-option judgments.

    Avoids the "Option Judge ≠ Final Aggregator" problem where the LLM
    correctly judges individual options but puts wrong answer in JSON.
    """
    if not results:
        return ""

    correct_opts = []
    weak_opts = []
    for r in results:
        judgment = str(r.get("judgment", ""))
        option = str(r.get("option", "")).upper()
        confidence = r.get("confidence", 0)
        if "正确" in judgment:
            if confidence >= 0.6:
                correct_opts.append(option)
            else:
                weak_opts.append(option)  # low-confidence "正确"

    if answer_format == "multi":
        # At least one correct: use all confirmed "正确"
        if correct_opts:
            return "".join(sorted(set(correct_opts)))
        # Nothing confirmed — accept weak positives
        if weak_opts:
            return "".join(sorted(set(weak_opts)))
        # All "错误" — but multi-choice must have ≥1 answer; pick lowest-confidence negatives
        scored = sorted(results, key=lambda r: r.get("confidence", 0))
        if scored:
            return str(scored[0].get("option", "")).upper()
        return ""

    else:  # mcq or tf — single choice
        if correct_opts:
            return correct_opts[0]
        if weak_opts:
            return weak_opts[0]
        # All "错误" → pick the least-confident "错误" (model is least sure about it)
        scored = sorted(results, key=lambda r: r.get("confidence", 0))
        if scored:
            return str(scored[0].get("option", "")).upper()
        return ""


def parse_reasoning_fallback(content: str, answer_format: str) -> dict:
    answer_patterns = [
        r'["\']?answer["\']?\s*[:：]\s*["\']?([A-Da-d]+)["\']?',
        r'答案\s*[是为：:]\s*([A-Da-d]+)',
        r'正确选项\s*[是为：:]\s*([A-Da-d]+)',
        r'答案为\s*([A-Da-d]+)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer_format == "multi":
                answer = "".join(sorted(set(answer)))
            return {"answer": answer, "results": [], "raw": content, "fallback": True}

    letters = re.findall(r'\b([A-D])\b', content)
    if letters:
        if answer_format == "multi":
            answer = "".join(sorted(set(letters)))
        else:
            answer = letters[0]
        return {"answer": answer, "results": [], "raw": content, "fallback": True}

    return {"answer": "", "results": [], "raw": content, "fallback": True}


def extract_facts(client: QwenClient, doc_id: str, doc_text: str,
                  question: str, domain: str) -> dict:
    """Round 1: extract structured facts + verification checkpoints from one doc."""
    prompt = f"""从以下文档中提取与问题相关的关键事实。

问题：{question}
文档来源：{doc_id}

文档内容：
{doc_text}

请输出两部分：

【已确认事实】（键值对，每条不超过30字）
- 指标名: 具体数值/条款文本

【需要验证的点】（文档中可能存在但未完全确认的细节，可能影响答案判断）
- 例：条款X是否有特殊情形下的例外？
- 例：数值Y的单位是万元还是亿元？

如果没有不确定的点，【需要验证的点】写"无"。
"""
    response = client.chat(
        messages=[build_chat_message("user", prompt)],
        temperature=0,
        max_tokens=500,
    )
    content = response["content"]
    # Parse out facts and verification points
    facts_section = ""
    verify_section = ""
    in_verify = False
    for line in content.split("\n"):
        line = line.strip()
        if "需要验证的点" in line or "不确定" in line:
            in_verify = True
            continue
        if in_verify:
            verify_section += line + "\n"
        else:
            facts_section += line + "\n"

    has_uncertainty = bool(verify_section.strip()) and "无" not in verify_section

    return {
        "facts": facts_section.strip()[:800],
        "verification_points": verify_section.strip() if has_uncertainty else "",
        "raw": content,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
    }


def reason_with_compression(client: QwenClient, evidence: list[dict],
                            question: str, options: dict, prompt_name: str,
                            answer_format: str, domain: str,
                            keyword_index: dict | None = None,
                            doc_ids: list[str] | None = None) -> dict:
    """Two-stage reasoning with domain-adaptive retry.

    Stage 1: Extract structured facts from each document.
    Stage 2: Reason from compressed facts.
    Retry strategy depends on domain:
    - insurance/financial_contracts: verification-point driven
    - financial_reports/research: confidence-gated
    """
    # ── Stage 1: Extract facts per document ──
    all_facts = []
    all_verification = []
    total_tokens = {"input": 0, "output": 0}

    for e in evidence:
        ef = extract_facts(client, e["doc_id"], e["text"], question, domain)
        all_facts.append(f"【{e['doc_id']}】\n{ef['facts']}")
        if ef["verification_points"]:
            all_verification.append(f"【{e['doc_id']}】\n{ef['verification_points']}")
        total_tokens["input"] += ef["input_tokens"]
        total_tokens["output"] += ef["output_tokens"]

    facts_context = "\n\n".join(all_facts)

    # ── Stage 2: Reason from compressed facts ──
    compressed_evidence = [{"doc_id": "_compressed_", "para_id": 0, "text": facts_context}]
    result = reason(client, compressed_evidence, question, options, prompt_name, answer_format)
    total_tokens["input"] += result.get("input_tokens", 0)
    total_tokens["output"] += result.get("output_tokens", 0)

    # ── Determine if retry needed ──
    needs_retry = False
    retry_reason = ""

    if domain in ("insurance", "financial_contracts"):
        # Verification-driven: retry if there are unverified points
        if all_verification:
            needs_retry = True
            retry_reason = f"verification: {len(all_verification)} docs have unresolved points"
    else:
        # Confidence-gated: retry if any option has low confidence
        from agent.validator import validate_confidence
        if not validate_confidence(result.get("results", [])):
            needs_retry = True
            retry_reason = "low confidence"

    # ── Retry with original evidence if needed ──
    if needs_retry:
        # Augment evidence with extracted facts as hints
        hints = "\n".join(all_facts)
        augmented_evidence = list(evidence)
        augmented_evidence.append({
            "doc_id": "_facts_hint_",
            "para_id": -1,
            "text": f"【已提取的关键事实】\n{hints}" + (f"\n\n【待验证的点】\n" + "\n".join(all_verification) if all_verification else "")
        })

        from agent.retriever import stage1_retrieve, expand_evidence
        if all_verification and keyword_index and doc_ids:
            # Supplementary retrieval for verification points
            verify_text = " ".join(all_verification)
            extra = stage1_retrieve(keyword_index, doc_ids, verify_text, options)
            if extra:
                augmented_evidence = expand_evidence(augmented_evidence + extra[:3], keyword_index)
            else:
                augmented_evidence = expand_evidence(augmented_evidence, keyword_index)
        else:
            augmented_evidence = expand_evidence(augmented_evidence, keyword_index or {})

        retry_result = reason(client, augmented_evidence[:10], question, options, prompt_name, answer_format)
        total_tokens["input"] += retry_result.get("input_tokens", 0)
        total_tokens["output"] += retry_result.get("output_tokens", 0)
        result = retry_result
        result["_retried"] = True
        result["_retry_reason"] = retry_reason

    result["input_tokens"] = total_tokens["input"]
    result["output_tokens"] = total_tokens["output"]
    result["_compressed"] = True
    return result


def reason(client: QwenClient, evidence: list[dict], question: str,
           options: dict, prompt_name: str, answer_format: str) -> dict:
    prompt = build_reasoning_prompt(prompt_name, evidence, question, options)

    response = client.chat(
        messages=[build_chat_message("user", prompt)],
        temperature=0,
        max_tokens=4096,
    )

    result = parse_reasoning_response(response["content"], answer_format)
    result["input_tokens"] = response["input_tokens"]
    result["output_tokens"] = response["output_tokens"]
    result["prompt_name"] = prompt_name
    return result
