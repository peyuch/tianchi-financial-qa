"""Run full pipeline on all 100 A榜 questions, generate answer.csv + evidence.json.

Usage:
  python scripts/run_all.py                          # default: mineru
  python scripts/run_all.py --pdf-backend pymupdf    # fast, plain text
  python scripts/run_all.py --pdf-backend mineru     # GPU accelerated
"""
import os, sys, json, io, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

from config import PROCESSED_DIR, OUTPUT_DIR, TOKEN_BUDGET, OUTPUT_CSV_FIELDNAMES, PDF_BACKEND
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import stage1_retrieve, should_skip_stage2, stage2_filter, allocate_per_doc
from agent.reasoner import reason, reason_with_compression

_COMPRESS_DOMAINS = {"insurance", "financial_contracts", "financial_reports"}

def _do_reason(client, evidence, question, options, prompt_name, af, domain,
               use_compress, keyword_index=None, doc_ids=None):
    if use_compress and domain in _COMPRESS_DOMAINS:
        return reason_with_compression(
            client, evidence, question, options, prompt_name, af, domain,
            keyword_index, doc_ids,
        )
    return reason(client, evidence, question, options, prompt_name, af)
from agent.validator import normalize_answer, validate_confidence, get_low_confidence_options, format_output_row, build_evidence_entry

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pdf-backend", default=PDF_BACKEND,
                    choices=["pymupdf4llm", "mineru", "pymupdf"],
                    help="PDF backend: pymupdf4llm (markdown,default), mineru (GPU), pymupdf (raw)")
parser.add_argument("--compress", action="store_true",
                    help="Enable two-stage fact extraction + compression reasoning")
args = parser.parse_args()
os.environ["PDF_BACKEND"] = args.pdf_backend
print(f"PDF backend: {args.pdf_backend} | Compress: {'ON' if args.compress else 'OFF'}")
print()

# Collect all questions
all_questions = []
question_dir = "public_dataset_upload/questions/group_a"
for fname in sorted(os.listdir(question_dir)):
    if fname.endswith(".json"):
        with open(os.path.join(question_dir, fname), "r", encoding="utf-8") as f:
            all_questions.extend(json.load(f))

print(f"Total questions: {len(all_questions)}")

# Collect all unique doc_ids
seen_docs = set()
for q in all_questions:
    for doc_id in q.get("doc_ids", []):
        seen_docs.add((q["domain"], doc_id))
print(f"Unique documents: {len(seen_docs)}")

client = QwenClient()
client.reset_counter()
start_time = time.time()

# === Phase 1: Preprocess all documents ===
print("\n" + "=" * 60)
print("PHASE 1: Preprocessing documents")
print("=" * 60)

doc_md_map = {}
for domain, doc_id in sorted(seen_docs):
    md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    if os.path.exists(md_path):
        doc_md_map[doc_id] = (domain, md_path)
        print(f"  [CACHED] {doc_id}")
        continue
    path = resolve_doc_path(domain, doc_id)
    if path is None:
        print(f"  [MISSING] {doc_id}")
        continue
    try:
        md_path = preprocess_document(path, os.path.splitext(path)[1])
        doc_md_map[doc_id] = (domain, md_path)
        print(f"  [OK] {doc_id}")
    except Exception as e:
        print(f"  [FAIL] {doc_id}: {e}")

# === Phase 2: Build indexes ===
print(f"\n{'=' * 60}")
print("PHASE 2: Building indexes")
print("=" * 60)

keyword_index = {}
for doc_id, (domain, md_path) in doc_md_map.items():
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        keyword_index.update(build_keyword_index(doc_id, text))
    except Exception as e:
        print(f"  [FAIL] {doc_id}: {e}")

print(f"Indexed {len(keyword_index)} documents")

# === Phase 3: Process questions ===
print(f"\n{'=' * 60}")
print("PHASE 3: Processing questions")
print("=" * 60)

answer_rows = []
evidence_entries = []
retry_count = 0
stage2_calls = 0

for i, q in enumerate(all_questions):
    qid = q["qid"]
    domain = q["domain"]
    question = q["question"]
    options = q["options"]
    answer_format = q["answer_format"]
    doc_ids = q.get("doc_ids", [])

    domain_config = route_domain(domain)
    prompt_name = select_reasoning_prompt(domain, question)

    # Snapshot token counter
    q_prompt_before = client.counter.prompt_tokens
    q_completion_before = client.counter.completion_tokens

    # Retrieve
    candidates = stage1_retrieve(keyword_index, doc_ids, question)
    skip_s2 = should_skip_stage2(candidates, question)

    if not skip_s2:
        try:
            evidence, s2_usage = stage2_filter(client, candidates, question, options)
            stage2_calls += 1
        except Exception as e:
            print(f"  [{i+1}/100] {qid}: S2 failed ({e}), fallback")
            evidence = candidates[:5]
    else:
        evidence = candidates[:5]

    # Per-doc allocation for insurance
    if domain == "insurance":
        evidence = allocate_per_doc(evidence, per_doc_cap=domain_config["per_doc_cap"], max_docs=domain_config["max_docs"])

    # Reason
    result = _do_reason(client, evidence, question, options, prompt_name, answer_format,
                        domain, args.compress, keyword_index, doc_ids)

    # Retry on low confidence
    if not validate_confidence(result.get("results", [])):
        low_opts = get_low_confidence_options(result.get("results", []))
        for opt in low_opts:
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
            unique_evidence = allocate_per_doc(unique_evidence, per_doc_cap=domain_config["per_doc_cap"], max_docs=domain_config["max_docs"])
        result = _do_reason(client, unique_evidence[:8], question, options, prompt_name, answer_format,
                            domain, args.compress, keyword_index, doc_ids)
        retry_count += 1

    # Normalize
    answer = normalize_answer(result.get("answer", ""), answer_format)
    result["answer"] = answer

    # Per-question token delta
    q_prompt = client.counter.prompt_tokens - q_prompt_before
    q_completion = client.counter.completion_tokens - q_completion_before

    elapsed = time.time() - start_time
    eta = (elapsed / (i + 1)) * (len(all_questions) - i - 1) if i > 0 else 0
    print(f"[{i+1:3d}/100] {qid} | ans={answer:4s} | prompt={prompt_name:20s} | "
          f"S2={'skip' if skip_s2 else 'call'} | "
          f"tok={q_prompt}/{q_completion} | "
          f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

    answer_rows.append(format_output_row(qid, answer, q_prompt, q_completion))
    evidence_entries.append(build_evidence_entry(qid, result))

# === Phase 4: Write output ===
print(f"\n{'=' * 60}")
print("PHASE 4: Writing output")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
import csv

total_prompt = client.counter.prompt_tokens
total_completion = client.counter.completion_tokens
total_tokens = total_prompt + total_completion

# answer.csv
answer_path = os.path.join(OUTPUT_DIR, "answer.csv")
with open(answer_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDNAMES)
    writer.writeheader()
    # Summary row
    writer.writerow({"qid": "summary", "answer": "", "prompt_tokens": total_prompt, "completion_tokens": total_completion, "total_tokens": total_tokens})
    # Per-question rows
    for row in answer_rows:
        writer.writerow(row)

# evidence.json
evidence_path = os.path.join(OUTPUT_DIR, "evidence.json")
with open(evidence_path, "w", encoding="utf-8") as f:
    json.dump(evidence_entries, f, ensure_ascii=False, indent=2)

# Budget report
print(f"\nanswer.csv -> {answer_path}")
print(f"evidence.json -> {evidence_path}")
print(f"\nTotal time: {time.time() - start_time:.0f}s")
print(f"Total tokens: {total_tokens:,} / {TOKEN_BUDGET:,}")
print(f"TokenScore: {max(0, min(1, (TOKEN_BUDGET - total_tokens) / TOKEN_BUDGET)):.3f}")
print(f"Stage 2 calls: {stage2_calls}/100")
print(f"Retries: {retry_count}")
print("DONE")
