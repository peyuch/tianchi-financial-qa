"""Round 2: test prefilter + metrics-anchored stage2 on 15 still-empty questions."""
import os, sys, json, io, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.indexer import build_keyword_index
from agent.domain_router import route_domain, select_reasoning_prompt
from agent.retriever import (stage1_retrieve, should_skip_stage2, stage2_filter,
                              allocate_per_doc, prefilter_candidates)
from agent.reasoner import reason
from agent.validator import normalize_answer

# The 15 still-empty QIDs from round 1
still_empty = ['fin_a_009', 'fc_a_007', 'fc_a_009', 'fc_a_004', 'fin_a_015',
               'fin_a_008', 'res_a_017', 'res_a_011', 'fin_a_020', 'ins_a_007',
               'ins_a_001', 'ins_a_002', 'fc_a_017', 'fin_a_005', 'fin_a_017']

# Load all questions
all_qs = {}
for fname in os.listdir('public_dataset_upload/questions/group_a'):
    if fname.endswith('.json'):
        with open(f'public_dataset_upload/questions/group_a/{fname}', 'r', encoding='utf-8') as f:
            for q in json.load(f):
                all_qs[q['qid']] = q

test_qs = [all_qs[qid] for qid in still_empty if qid in all_qs]
print(f'Testing {len(test_qs)} still-empty questions\n')

client = QwenClient()
fixed = 0
still_empty_count = 0

for i, q in enumerate(test_qs):
    qid, domain, question = q['qid'], q['domain'], q['question']
    options, af, doc_ids = q['options'], q['answer_format'], q['doc_ids']

    keyword_index = {}
    for doc_id in doc_ids:
        md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                keyword_index.update(build_keyword_index(doc_id, f.read()))

    candidates = stage1_retrieve(keyword_index, doc_ids, question, options)
    n_before = len(candidates)
    candidates = prefilter_candidates(candidates, domain)
    n_after = len(candidates)

    domain_config = route_domain(domain)
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

    prompt_name = select_reasoning_prompt(domain, question)
    result = reason(client, evidence, question, options, prompt_name, af)
    answer = normalize_answer(result.get('answer', ''), af)

    status = 'FIXED' if answer else 'STILL EMPTY'
    if answer:
        fixed += 1
    else:
        still_empty_count += 1

    print(f'  [{i+1:2d}/15] {qid:10s} | ans={answer:4s} | {status:12s} | '
          f'S1={n_before}→pre={n_after}→S2={len(evidence)} | '
          f'tok={result.get("input_tokens",0)}')

print(f'\nRound 2: Fixed {fixed}/15, Still empty {still_empty_count}/15')
