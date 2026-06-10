"""Test pipeline on a single regulatory question — encoding-safe."""
import os, sys, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import RAW_DIR, PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import stage1_retrieve, should_skip_stage2, stage2_filter
from agent.reasoner import reason
from agent.validator import normalize_answer

# Load first regulatory true/false question
with open("public_dataset_upload/questions/group_a/regulatory_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

q = questions[5]  # reg_a_006 — true/false
qid = q["qid"]
domain = q["domain"]
question = q["question"]
options = q["options"]
answer_format = q["answer_format"]
doc_ids = q["doc_ids"]

print(f"Q: {qid} ({domain}, {answer_format})")
print(f"Question: {question[:150]}")
print(f"Options: {options}")
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
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    doc_idx = build_keyword_index(doc_id, text)
    keyword_index.update(doc_idx)
    print(f"  Paragraphs: {len(doc_idx.get(doc_id, []))}")
    print()

# 2. Retrieve
candidates = stage1_retrieve(keyword_index, doc_ids, question)
print(f"Stage 1: {len(candidates)} candidates")
skip_s2 = should_skip_stage2(candidates, question)
print(f"Skip S2: {skip_s2}")
if not skip_s2:
    evidence, s2_usage = stage2_filter(client, candidates, question, options)
else:
    evidence = candidates[:5]
print(f"Evidence: {len(evidence)} paragraphs")
print()

# 3. Reason
prompt_name = select_reasoning_prompt(domain, question)
print(f"Prompt: {prompt_name}")
print("Calling Qwen API for reasoning...")
result = reason(client, evidence, question, options, prompt_name, answer_format)

answer = normalize_answer(result.get("answer", ""), answer_format)
print(f"\nAnswer: {answer}")
print(f"Raw answer from model: {result.get('answer', '')}")
print(f"Results: {json.dumps(result.get('results', []), ensure_ascii=False, indent=2)[:500]}")
print(f"Tokens: in={result['input_tokens']} out={result['output_tokens']}")
print(f"Total: {client.counter.total_tokens}")
