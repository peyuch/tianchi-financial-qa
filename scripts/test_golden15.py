"""Run 15 golden questions — first 3 from each domain.

Usage:
  python scripts/test_golden15.py                          # default: mineru
  python scripts/test_golden15.py --pdf-backend pymupdf    # fast, plain text
  python scripts/test_golden15.py --pdf-backend mineru     # GPU accelerated
"""
import os, sys, json, io, time, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR, OUTPUT_CSV_FIELDNAMES, PDF_BACKEND
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import (
    stage1_retrieve, should_skip_stage2, stage2_filter,
    allocate_per_doc, prefilter_candidates, expand_evidence,
)
from agent.reasoner import reason, reason_with_compression

_COMPRESS_DOMAINS = {"insurance", "financial_contracts", "financial_reports"}

def _do_reason(client, evidence, question, options, prompt_name, af, domain,
               use_compress, keyword_index=None, doc_ids=None):
    """Reason with optional two-stage compression."""
    if use_compress and domain in _COMPRESS_DOMAINS:
        return reason_with_compression(
            client, evidence, question, options, prompt_name, af, domain,
            keyword_index, doc_ids,
        )
    return reason(client, evidence, question, options, prompt_name, af)
from agent.validator import (normalize_answer, validate_confidence,
                              get_low_confidence_options, build_evidence_entry)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pdf-backend", default=PDF_BACKEND, choices=["mineru", "pymupdf"],
                    help="PDF parsing backend (default: %(default)s)")
parser.add_argument("--compress", action="store_true",
                    help="Enable two-stage fact extraction + compression reasoning")
args = parser.parse_args()
os.environ["PDF_BACKEND"] = args.pdf_backend
print(f"PDF backend: {args.pdf_backend} | Compress: {'ON' if args.compress else 'OFF'}")
print()

# Load golden questions
with open("tests/golden_15.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Golden 15 questions ({len(questions)} total):")
for q in questions:
    print(f"  {q['qid']} [{q['domain']}] {q['answer_format']} — {q['question'][:80]}...")
print()

# Collect unique docs and preprocess
seen_docs = set()
for q in questions:
    for doc_id in q.get("doc_ids", []):
        seen_docs.add((q["domain"], doc_id))

print(f"Documents needed: {len(seen_docs)}")
client = QwenClient()
client.reset_counter()

# Preprocess + index
keyword_index = {}
i = 0
for domain, doc_id in sorted(seen_docs):
    i += 1
    md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    if os.path.exists(md_path):
        print(f"  [{i}/{len(seen_docs)}] {doc_id} (cached)")
    else:
        print(f"  [{i}/{len(seen_docs)}] {doc_id} (parsing, backend={args.pdf_backend}...)", flush=True)
        path = resolve_doc_path(domain, doc_id)
        if path:
            try:
                md_path = preprocess_document(path, os.path.splitext(path)[1])
                print(f"    -> OK ({os.path.getsize(md_path)} bytes)")
            except Exception as e:
                print(f"    -> FAIL: {e}")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            keyword_index.update(build_keyword_index(doc_id, f.read(), domain, args.pdf_backend))

print(f"Indexed {len(keyword_index)} documents\n")

# Process questions
results = []
evidence_entries = []
start = time.time()

for i, q in enumerate(questions):
    qid, domain = q["qid"], q["domain"]
    question, options = q["question"], q["options"]
    af, doc_ids = q["answer_format"], q.get("doc_ids", [])

    domain_config = route_domain(domain)
    prompt_name = select_reasoning_prompt(domain, question)

    q_prompt_before = client.counter.prompt_tokens
    q_completion_before = client.counter.completion_tokens

    # Retrieve
    candidates = stage1_retrieve(keyword_index, doc_ids, question, options)
    candidates = prefilter_candidates(candidates, domain)
    skip_s2 = should_skip_stage2(candidates, question)

    if not skip_s2:
        try:
            evidence, _ = stage2_filter(client, candidates, question, options, doc_ids)
        except:
            evidence = candidates[:5]
    else:
        evidence = candidates[:5]

    evidence = expand_evidence(evidence, keyword_index, expand_by=1)
    if domain == "insurance":
        evidence = allocate_per_doc(evidence, per_doc_cap=domain_config["per_doc_cap"], max_docs=domain_config["max_docs"])

    # Reason
    result = _do_reason(client, evidence, question, options, prompt_name, af,
                        domain, args.compress, keyword_index, doc_ids)

    # Retry on low confidence
    if not validate_confidence(result.get("results", [])):
        low_opts = get_low_confidence_options(result.get("results", []))
        for opt in low_opts:
            extra = stage1_retrieve(keyword_index, doc_ids, options.get(opt, ""), options)
            evidence.extend(extra[:3])
        seen_texts = set()
        unique_evidence = []
        for e in evidence:
            if e["text"] not in seen_texts:
                seen_texts.add(e["text"])
                unique_evidence.append(e)
        if domain == "insurance":
            unique_evidence = allocate_per_doc(unique_evidence, per_doc_cap=domain_config["per_doc_cap"], max_docs=domain_config["max_docs"])
        result = _do_reason(client, unique_evidence[:8], question, options, prompt_name, af,
                            domain, args.compress, keyword_index, doc_ids)

    answer = normalize_answer(result.get("answer", ""), af)

    q_prompt = client.counter.prompt_tokens - q_prompt_before
    q_completion = client.counter.completion_tokens - q_completion_before

    elapsed = time.time() - start
    print(f"[{i+1:2d}/15] {qid} | ans={answer:4s} | {prompt_name:22s} | "
          f"S2={'skip' if skip_s2 else 'call'} | tok={q_prompt}/{q_completion} | {elapsed:.0f}s")

    result["answer"] = answer
    evidence_entries.append(build_evidence_entry(qid, result))
    results.append({
        "qid": qid,
        "domain": domain,
        "answer": answer,
        "prompt_tokens": q_prompt,
        "completion_tokens": q_completion,
        "total_tokens": q_prompt + q_completion,
        "prompt_name": prompt_name,
    })

# Summary
print(f"\n{'='*60}")
print(f"Total time: {time.time() - start:.0f}s")
print(f"Total tokens: {client.counter.total_tokens}")
print(f"\nResults:")
for r in results:
    print(f"  {r['qid']} [{r['domain']:20s}] → {r['answer']:4s}  ({r['total_tokens']} tok)")

# Write CSV
import csv
os.makedirs("output", exist_ok=True)
with open("output/golden15_answer.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerow({"qid": "summary", "answer": "", "prompt_tokens": client.counter.prompt_tokens, "completion_tokens": client.counter.completion_tokens, "total_tokens": client.counter.total_tokens})
    for r in results:
        writer.writerow({"qid": r["qid"], "answer": r["answer"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "total_tokens": r["total_tokens"]})
print("\nWrote output/golden15_answer.csv")

with open("output/golden15_evidence.json", "w", encoding="utf-8") as f:
    json.dump(evidence_entries, f, ensure_ascii=False, indent=2)
print("Wrote output/golden15_evidence.json")
