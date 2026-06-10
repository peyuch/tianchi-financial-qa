"""Test full pipeline with ModelScope backend."""
import os, sys, json, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["QWEN_BACKEND"] = "modelscope"
os.environ["MODELSCOPE_API_KEY"] = "ms-165a851e-841f-4744-b949-892d0fa38251"
os.environ["QWEN_MODEL"] = "Qwen/Qwen3.5-35B-A3B"

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import stage1_retrieve, should_skip_stage2, stage2_filter
from agent.reasoner import reason
from agent.validator import normalize_answer

# reg_a_006 — simple true/false question
with open("public_dataset_upload/questions/group_a/regulatory_questions.json", "r", encoding="utf-8") as f:
    qs = json.load(f)
q = qs[5]

qid, domain, question = q["qid"], q["domain"], q["question"]
options, af, doc_ids = q["options"], q["answer_format"], q["doc_ids"]

print(f"Q: {qid} ({af}) — {question[:120]}")
print()

client = QwenClient()

# Preprocess + index
keyword_index = {}
for doc_id in doc_ids:
    md_path = os.path.join(PROCESSED_DIR, f"{doc_id}.md")
    if not os.path.exists(md_path):
        path = resolve_doc_path(domain, doc_id)
        md_path = preprocess_document(path, os.path.splitext(path)[1])
    with open(md_path, "r", encoding="utf-8") as f:
        keyword_index.update(build_keyword_index(doc_id, f.read()))

# Retrieve
candidates = stage1_retrieve(keyword_index, doc_ids, question)
skip_s2 = should_skip_stage2(candidates, question)
if not skip_s2:
    evidence, _ = stage2_filter(client, candidates, question, options)
else:
    evidence = candidates[:5]

print(f"Candidates: {len(candidates)} → Evidence: {len(evidence)} (S2 skip: {skip_s2})")

# Reason
prompt_name = select_reasoning_prompt(domain, question)
result = reason(client, evidence, question, options, prompt_name, af)
answer = normalize_answer(result.get("answer", ""), af)

print(f"Answer: {answer} | Prompt: {prompt_name}")
print(f"Tokens: in={result['input_tokens']} out={result['output_tokens']}")
print(f"Total: {client.counter.total_tokens}")
