"""Deep trace: for each wrong question, dump full pipeline state for analysis."""
import sys, io, json, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.indexer import build_keyword_index
from agent.domain_router import select_reasoning_prompt, route_domain
from agent.retriever import (
    stage1_retrieve, prefilter_candidates, stage2_filter, expand_evidence,
    extract_keywords_from_question,
)
from agent.reasoner import reason, build_reasoning_prompt
from agent.validator import normalize_answer

wrong_qids = ['fc_a_001','fc_a_002','fin_a_001','fin_a_002','reg_a_001','reg_a_002','res_a_001','res_a_002']
std_answers = {
    'fc_a_001':'AB','fc_a_002':'ABD','fin_a_001':'AB','fin_a_002':'ABD',
    'reg_a_001':'ACD','reg_a_002':'ABC','res_a_001':'ABC','res_a_002':'ABCD'
}

with open('tests/golden_15.json', 'r', encoding='utf-8') as f:
    qs = json.load(f)

# Build index once
keyword_index = {}
all_qs = {}
for q in qs:
    all_qs[q['qid']] = q
    for doc_id in q.get('doc_ids', []):
        md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
        if os.path.exists(md_path) and doc_id not in keyword_index:
            with open(md_path, 'r', encoding='utf-8') as f:
                keyword_index.update(build_keyword_index(doc_id, f.read(), q['domain'], 'pymupdf4llm'))

client = QwenClient()

# Run with full trace
trace_report = []
correct, wrong = 0, 0

for qid in wrong_qids:
    q = all_qs[qid]
    domain, question, options = q['domain'], q['question'], q['options']
    af, doc_ids = q['answer_format'], q['doc_ids']
    std = std_answers[qid]

    trace = {
        'qid': qid, 'domain': domain, 'question': question,
        'options': options, 'doc_ids': doc_ids, 'std': std
    }

    # Stage 1
    keywords = extract_keywords_from_question(question)
    candidates = stage1_retrieve(keyword_index, doc_ids, question, options)
    trace['s1_keywords'] = keywords[:15]
    trace['s1_candidates'] = len(candidates)
    # Per-doc breakdown
    from collections import Counter
    trace['s1_per_doc'] = dict(Counter(c['doc_id'] for c in candidates))

    # Prefilter
    before_pf = len(candidates)
    candidates = prefilter_candidates(candidates, domain)
    trace['s1_after_prefilter'] = len(candidates)
    trace['s1_filtered_out'] = before_pf - len(candidates)

    # Stage 2
    domain_config = route_domain(domain)
    evidence, s2_usage = stage2_filter(client, candidates, question, options, doc_ids)
    trace['s2_evidence_count'] = len(evidence)
    trace['s2_per_doc'] = dict(Counter(e['doc_id'] for e in evidence))
    trace['s2_tokens'] = s2_usage

    # Evidence details (first 3)
    trace['s2_evidence_preview'] = []
    for e in evidence[:5]:
        trace['s2_evidence_preview'].append({
            'doc_id': e['doc_id'][:30],
            'text_preview': e['text'][:200].replace('\n',' '),
            'char_len': len(e['text']),
        })

    # Expand
    evidence = expand_evidence(evidence, keyword_index)
    trace['s2_after_expand'] = len(evidence)

    # Build prompt (truncate for display)
    prompt_name = select_reasoning_prompt(domain, question)
    full_prompt = build_reasoning_prompt(prompt_name, evidence, question, options)
    trace['prompt_name'] = prompt_name
    trace['prompt_len_chars'] = len(full_prompt)
    trace['prompt_preview'] = full_prompt[:800]

    # Reason
    result = reason(client, evidence, question, options, prompt_name, af)
    answer = normalize_answer(result.get('answer', ''), af)
    trace['model_answer'] = answer
    trace['model_raw_first_800'] = result.get('raw','')[:800]
    trace['tokens_in'] = result.get('input_tokens',0)
    trace['tokens_out'] = result.get('output_tokens',0)

    # Option judgments
    trace['option_judgments'] = []
    for r in result.get('results', []):
        trace['option_judgments'].append({
            'option': r.get('option'),
            'judgment': r.get('judgment'),
            'confidence': r.get('confidence'),
            'evidence_doc': r.get('evidence',{}).get('doc_id','')[:30],
            'reasoning': r.get('evidence',{}).get('reasoning','')[:300],
        })

    match = '✅' if answer == std else '❌'
    if answer == std: correct += 1
    else: wrong += 1
    trace['match'] = match

    print(f"{qid}: {answer} vs {std} {match} | S1:{trace['s1_candidates']}->S2:{trace['s2_evidence_count']} |")
    trace_report.append(trace)

# Write trace file
with open('output/trace_wrong_questions.json', 'w', encoding='utf-8') as f:
    json.dump(trace_report, f, ensure_ascii=False, indent=2)

# Now read back and check source docs for missing evidence
print(f"\n{correct}/8 correct")
print(f"Full trace written to output/trace_wrong_questions.json")
print(f"\n=== Source document evidence check ===")
for trace in trace_report:
    qid = trace['qid']
    std_set = set(std_answers[qid])
    model_set = set(trace['model_answer'])
    missed = std_set - model_set
    if missed:
        print(f"\n--- {qid}: missed options {missed} ---")
        for opt in missed:
            opt_text = trace['options'][opt]
            print(f"  [{opt}] {opt_text[:100]}")
            # Check source docs
            for doc_id in trace['doc_ids']:
                md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        doc_text = f.read()
                    # Check key terms
                    terms = re.findall(r'[一-鿿]{2,}|\d+[万亿千百]?', opt_text)
                    found = [t for t in terms[:8] if t in doc_text and len(t) > 1]
                    if found:
                        print(f"    {doc_id}: TERMS IN DOC -> {found[:5]}")
                    # Check if terms also in evidence
                    ev_text = ' '.join(e['text_preview'] for e in trace.get('s2_evidence_preview',[]))
                    in_ev = [t for t in found if t in ev_text]
                    if in_ev:
                        print(f"      -> IN EVIDENCE: {in_ev[:5]}")
                    else:
                        print(f"      -> NOT IN EVIDENCE (retrieval gap!)")
