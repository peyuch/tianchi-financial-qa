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
    json_match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', content)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return {
                "answer": data.get("answer", ""),
                "results": data.get("results", []),
                "raw": content,
            }
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(content.strip())
        return {
            "answer": data.get("answer", ""),
            "results": data.get("results", []),
            "raw": content,
        }
    except json.JSONDecodeError:
        return parse_reasoning_fallback(content, answer_format)


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


def reason(client: QwenClient, evidence: list[dict], question: str,
           options: dict, prompt_name: str, answer_format: str) -> dict:
    prompt = build_reasoning_prompt(prompt_name, evidence, question, options)

    response = client.chat(
        messages=[build_chat_message("user", prompt)],
        temperature=0.1,
        max_tokens=4096,
    )

    result = parse_reasoning_response(response["content"], answer_format)
    result["input_tokens"] = response["input_tokens"]
    result["output_tokens"] = response["output_tokens"]
    result["prompt_name"] = prompt_name
    return result
