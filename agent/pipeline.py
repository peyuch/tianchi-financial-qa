"""Pipeline: main orchestration — wires all modules together."""
import json
import os
import csv
from config import (
    QUESTIONS_DIR, PROCESSED_DIR, INDEX_DIR, SUMMARIES_DIR, OUTPUT_DIR,
    TOKEN_BUDGET, OUTPUT_CSV_FIELDNAMES,
)
from agent.qwen_client import QwenClient, TokenCounter
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index, extract_summary, search_keyword_index, search_multi_keyword
from agent.domain_router import route_domain, select_reasoning_prompt, get_output_format
from agent.retriever import (
    stage1_retrieve, stage2_filter, should_skip_stage2,
    extract_keywords_from_question, allocate_per_doc, prefilter_candidates,
    expand_evidence,
)
from agent.reasoner import reason
from agent.validator import (
    normalize_answer, validate_confidence,
    get_low_confidence_options, format_output_row, build_evidence_entry,
    check_shared_evidence_risk,
)


def load_questions(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_doc_list(questions: list[dict]) -> list[tuple[str, str]]:
    seen = set()
    pairs = []
    for q in questions:
        domain = q["domain"]
        for doc_id in q.get("doc_ids", []):
            key = (domain, doc_id)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def run_pipeline(
    question_file: str,
    client: QwenClient,
    skip_preprocess: bool = False,
    domain_filter: str | None = None,
) -> tuple[str, str, TokenCounter]:
    client.reset_counter()
    questions = load_questions(question_file)

    if domain_filter:
        questions = [q for q in questions if q["domain"] == domain_filter]

    # Step 1: Preprocess documents
    doc_pairs = collect_doc_list(questions)
    print(f"\n=== Preprocessing {len(doc_pairs)} documents ===")

    doc_md_map = {}
    if not skip_preprocess:
        for domain, doc_id in doc_pairs:
            path = resolve_doc_path(domain, doc_id)
            if path is None:
                print(f"  SKIP: {doc_id} not found")
                continue
            ext = os.path.splitext(path)[1]
            try:
                md_path = preprocess_document(path, ext)
                doc_md_map[doc_id] = (domain, md_path)
                print(f"  OK: {doc_id}")
            except Exception as e:
                print(f"  FAIL: {doc_id}: {e}")
    else:
        for domain, doc_id in doc_pairs:
            md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
            if os.path.isfile(md_path):
                doc_md_map[doc_id] = (domain, md_path)

    # Step 2: Build indexes
    print(f"\n=== Building indexes for {len(doc_md_map)} documents ===")
    keyword_index = {}
    summaries = []

    for doc_id, (domain, md_path) in doc_md_map.items():
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        pdf_backend = os.environ.get("PDF_BACKEND", "pymupdf")
        doc_index = build_keyword_index(doc_id, text, domain, pdf_backend)
        keyword_index.update(doc_index)
        summary = extract_summary(doc_id, domain, text)
        summaries.append(summary)

    print(f"  Indexed {len(keyword_index)} documents")
    print(f"  Generated {len(summaries)} summaries")

    # Step 3: Process each question
    print(f"\n=== Processing {len(questions)} questions ===")
    answer_rows = []
    evidence_entries = []
    retry_count = 0

    for i, q in enumerate(questions):
        qid = q["qid"]
        domain = q["domain"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]
        doc_ids = q.get("doc_ids", [])

        domain_config = route_domain(domain)
        prompt_name = select_reasoning_prompt(domain, question)
        output_rules = get_output_format(answer_format)

        q_prompt_before = client.counter.prompt_tokens
        q_completion_before = client.counter.completion_tokens

        print(f"\n[{i+1}/{len(questions)}] {qid} ({domain}, {answer_format})")

        # Retrieve
        candidates = stage1_retrieve(keyword_index, doc_ids, question, options)
        candidates = prefilter_candidates(candidates, domain)

        skip_s2 = should_skip_stage2(candidates, question)
        s2_usage = {"input_tokens": 0, "output_tokens": 0}
        if not skip_s2:
            try:
                evidence, s2_usage = stage2_filter(client, candidates, question, options, doc_ids)
            except Exception as e:
                print(f"  WARNING: Stage 2 failed ({e}), falling back to top-5 candidates")
                evidence = candidates[:5]
        else:
            evidence = candidates[:5]

        # Expand evidence: for each selected paragraph, also include its next paragraph
        evidence = expand_evidence(evidence, keyword_index, expand_by=1)

        # Apply per-doc allocation for insurance domain
        if domain == "insurance":
            evidence = allocate_per_doc(
                evidence,
                per_doc_cap=domain_config["per_doc_cap"],
                max_docs=domain_config["max_docs"],
            )

        # Reason
        result = reason(client, evidence, question, options, prompt_name, answer_format)

        # Validate
        needs_retry = False
        if not validate_confidence(result.get("results", [])):
            needs_retry = True
        elif check_shared_evidence_risk(result.get("results", [])):
            print("  Shared evidence risk detected, retrying per-option...")
            needs_retry = True

        if needs_retry:
            # Shared-evidence → retry ALL options; low-confidence → retry only weak ones
            if check_shared_evidence_risk(result.get("results", [])):
                retry_opts = list(options.keys())
                reason_label = "shared-evidence"
            else:
                retry_opts = get_low_confidence_options(result.get("results", []))
                reason_label = "low-confidence"
            print(f"  Retry ({reason_label}) on: {retry_opts}")

            for opt in retry_opts:
                opt_text = options.get(opt, "")
                extra = stage1_retrieve(keyword_index, doc_ids, opt_text)
                evidence.extend(extra[:3])

            seen_texts = set()
            unique_evidence = []
            for e in evidence:
                if e["text"] not in seen_texts:
                    seen_texts.add(e["text"])
                    unique_evidence.append(e)

            if domain == "insurance":
                unique_evidence = allocate_per_doc(
                    unique_evidence,
                    per_doc_cap=domain_config["per_doc_cap"],
                    max_docs=domain_config["max_docs"],
                )

            result = reason(client, unique_evidence[:8], question, options,
                          prompt_name, answer_format)
            retry_count += 1

        # Normalize answer
        answer = normalize_answer(result.get("answer", ""), answer_format)
        result["answer"] = answer

        q_prompt = client.counter.prompt_tokens - q_prompt_before
        q_completion = client.counter.completion_tokens - q_completion_before

        print(f"  Answer: {answer} | Prompt: {prompt_name} | "
              f"S2 skipped: {skip_s2} | Tokens: {q_prompt}/{q_completion}")

        answer_rows.append(format_output_row(qid, answer, q_prompt, q_completion))
        evidence_entries.append(build_evidence_entry(qid, result))

    # Step 4: Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    answer_path = os.path.join(OUTPUT_DIR, "answer.csv")
    with open(answer_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDNAMES)
        writer.writeheader()

        total_prompt = client.counter.prompt_tokens
        total_completion = client.counter.completion_tokens
        writer.writerow({
            "qid": "summary",
            "answer": "",
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        })

        for row in answer_rows:
            writer.writerow(row)

    evidence_path = os.path.join(OUTPUT_DIR, "evidence.json")
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_entries, f, ensure_ascii=False, indent=2)

    # Budget report
    total = total_prompt + total_completion
    budget_remaining = TOKEN_BUDGET - total
    print(f"\n=== Budget Report ===")
    print(f"  Total tokens: {total:,} / {TOKEN_BUDGET:,}")
    print(f"  Remaining: {budget_remaining:,}")
    print(f"  Retries: {retry_count}")
    print(f"  TokenScore: {max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET)):.3f}")

    return answer_path, evidence_path, client.counter
