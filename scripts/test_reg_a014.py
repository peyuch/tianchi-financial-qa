"""Test pipeline on reg_a_014 — the introduct.md example question.
Standard answer: AC
"""
import os, sys, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import stage1_retrieve, should_skip_stage2, stage2_filter
from agent.reasoner import reason
from agent.validator import normalize_answer

# Load reg_a_014
with open("public_dataset_upload/questions/group_a/regulatory_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

q = None
for qq in questions:
    if qq["qid"] == "reg_a_014":
        q = qq
        break

qid = q["qid"]
domain = q["domain"]
question = q["question"]
options = q["options"]
answer_format = q["answer_format"]
doc_ids = q["doc_ids"]

print("=" * 60)
print(f"Q: {qid} ({domain}, {answer_format})")
print(f"Question: {question}")
print(f"Options:")
for k, v in sorted(options.items()):
    print(f"  {k}. {v}")
print(f"Docs: {doc_ids}")
print(f"Standard answer: AC")
print("=" * 60)
print()

# 1. Preprocess + index
client = QwenClient()
keyword_index = {}

for doc_id in doc_ids:
    path = resolve_doc_path(domain, doc_id)
    print(f"Doc: {doc_id}")
    print(f"  Path: {path}")
    md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    if not os.path.exists(md_path):
        ext = os.path.splitext(path)[1]
        md_path = preprocess_document(path, ext)
        print(f"  Preprocessed => {md_path}")
    else:
        print(f"  Already processed => {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    doc_idx = build_keyword_index(doc_id, text)
    keyword_index.update(doc_idx)
    n_paras = len(doc_idx.get(doc_id, []))
    print(f"  Paragraphs: {n_paras}")
    print(f"  Text length: {len(text)} chars")
    print()

# 2. Retrieve
print("--- Retrieval ---")
candidates = stage1_retrieve(keyword_index, doc_ids, question)
print(f"Stage 1 candidates: {len(candidates)}")

skip_s2 = should_skip_stage2(candidates, question)
print(f"Skip Stage 2: {skip_s2}")

if not skip_s2:
    print("Running Stage 2 Qwen filter...")
    evidence, s2_usage = stage2_filter(client, candidates, question, options)
    print(f"  Stage 2 tokens: {s2_usage}")
else:
    evidence = candidates[:5]
    s2_usage = {"input_tokens": 0, "output_tokens": 0}

print(f"Evidence paragraphs: {len(evidence)}")
for i, e in enumerate(evidence):
    text_preview = e['text'][:120].replace('\n', ' ')
    print(f"  [{i}] [{e['doc_id']}] {text_preview}...")
print()

# 3. Reason
prompt_name = select_reasoning_prompt(domain, question)
print(f"Reasoning prompt: {prompt_name}")
print("Calling Qwen API...")
result = reason(client, evidence, question, options, prompt_name, answer_format)

answer = normalize_answer(result.get("answer", ""), answer_format)
print(f"\n{'=' * 60}")
print(f"Model answer: {answer}")
print(f"Standard:     AC")
print(f"Match: {'YES' if answer == 'AC' else 'NO'}")
print(f"Tokens: in={result['input_tokens']} out={result['output_tokens']}")

# Show reasoning per option
for r in result.get("results", []):
    ev = r.get("evidence", {})
    print(f"\n  Option {r['option']}: {r['judgment']} (confidence: {r['confidence']})")
    print(f"    Doc: {ev.get('doc_id', '')}")
    print(f"    Reason: {ev.get('reasoning', '')[:200]}")
