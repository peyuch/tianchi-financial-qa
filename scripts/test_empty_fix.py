"""Test fix: re-run the 25 previously-empty questions after orphan-number merge."""
import os, sys, json, io, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.preprocessor import resolve_doc_path, preprocess_document
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import stage1_retrieve, should_skip_stage2, stage2_filter, allocate_per_doc
from agent.reasoner import reason
from agent.validator import normalize_answer, validate_confidence, get_low_confidence_options

# Get empty QIDs
empty_qids = set()
with open('D:/TianChi/output/answer.csv', 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        if not row['answer'].strip() and row['qid'] != 'summary':
            empty_qids.add(row['qid'])

# Load all questions
all_qs = {}
for fname in os.listdir('public_dataset_upload/questions/group_a'):
    if fname.endswith('.json'):
        with open(f'public_dataset_upload/questions/group_a/{fname}', 'r', encoding='utf-8') as f:
            for q in json.load(f):
                all_qs[q['qid']] = q

# Filter to empty questions
test_qs = [all_qs[qid] for qid in empty_qids if qid in all_qs]
print(f'Testing {len(test_qs)} previously-empty questions\n')

# Preprocess needed documents (force re-process since we deleted cache)
needed_docs = set()
for q in test_qs:
    for doc_id in q.get('doc_ids', []):
        needed_docs.add((q['domain'], doc_id))

print(f'Re-processing {len(needed_docs)} documents...')
for domain, doc_id in sorted(needed_docs):
    path = resolve_doc_path(domain, doc_id)
    if path:
        md_path = preprocess_document(path, os.path.splitext(path)[1])
        print(f'  OK: {doc_id}')
    else:
        print(f'  MISS: {doc_id}')

# Run questions
client = QwenClient()
fixed = 0
still_empty = 0

for i, q in enumerate(test_qs):
    qid, domain, question = q['qid'], q['domain'], q['question']
    options, af, doc_ids = q['options'], q['answer_format'], q['doc_ids']

    # Build index
    keyword_index = {}
    for doc_id in doc_ids:
        md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                keyword_index.update(build_keyword_index(doc_id, f.read()))

    # Retrieve
    domain_config = route_domain(domain)
    candidates = stage1_retrieve(keyword_index, doc_ids, question)
    skip_s2 = should_skip_stage2(candidates, question)

    if not skip_s2:
        try:
            evidence, _ = stage2_filter(client, candidates, question, options)
        except:
            evidence = candidates[:5]
    else:
        evidence = candidates[:5]

    if domain == 'insurance':
        evidence = allocate_per_doc(evidence, per_doc_cap=domain_config['per_doc_cap'], max_docs=domain_config['max_docs'])

    # Reason
    prompt_name = select_reasoning_prompt(domain, question)
    result = reason(client, evidence, question, options, prompt_name, af)
    answer = normalize_answer(result.get('answer', ''), af)

    status = 'FIXED' if answer else 'STILL EMPTY'
    if answer:
        fixed += 1
    else:
        still_empty += 1

    # Check evidence quality
    has_numbers = sum(1 for e in evidence if any(c.isdigit() for c in e['text']))

    print(f'  [{i+1:2d}/25] {qid} | ans={answer:4s} | {status:12s} | '
          f'ev={len(evidence)} | num_ev={has_numbers} | tok={result.get("input_tokens",0)}')

print(f'\nFixed: {fixed}/25, Still empty: {still_empty}/25')
