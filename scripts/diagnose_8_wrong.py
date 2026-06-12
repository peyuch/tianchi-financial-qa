"""Full-chain diagnostic report for 8 historically wrong questions.

Generates: output/diagnostic_report.md
"""
import sys, io, json, os, re, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import PROCESSED_DIR
from agent.qwen_client import QwenClient
from agent.indexer import build_keyword_index
from agent.domain_router import select_reasoning_prompt, route_domain
from agent.retriever import (
    stage1_retrieve, prefilter_candidates, stage2_filter, expand_evidence,
    extract_keywords_from_question, allocate_per_doc,
)
from agent.reasoner import reason, build_reasoning_prompt
from agent.validator import normalize_answer

WRONG_QIDS = ['fc_a_001','fc_a_002','fin_a_001','fin_a_002',
              'reg_a_001','reg_a_002','res_a_001','res_a_002']
STD = {
    'fc_a_001':'AB','fc_a_002':'ABD','fin_a_001':'AB','fin_a_002':'ABD',
    'reg_a_001':'ACD','reg_a_002':'ABC','res_a_001':'ABC','res_a_002':'ABCD'
}

with open('tests/golden_15.json', 'r', encoding='utf-8') as f:
    all_qs = {q['qid']: q for q in json.load(f)}

# Build index once
keyword_index = {}
for qid in WRONG_QIDS:
    q = all_qs[qid]
    for doc_id in q.get('doc_ids', []):
        md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
        if os.path.exists(md_path) and doc_id not in keyword_index:
            with open(md_path, 'r', encoding='utf-8') as f:
                keyword_index.update(build_keyword_index(
                    doc_id, f.read(), q['domain'], 'pymupdf4llm'))

client = QwenClient()

md_lines = [
    "# 🔬 Golden15 错题全链路诊断报告",
    f"\n> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 模型: qwen-max temp=0 | 后端: pymupdf4llm\n",
]
correct_count = 0

for qid in WRONG_QIDS:
    q = all_qs[qid]
    domain = q['domain']
    question = q['question']
    options = q['options']
    af = q['answer_format']
    doc_ids = q['doc_ids']
    std = STD[qid]

    md_lines.append(f"---\n## {qid}")
    md_lines.append(f"\n**题目**: {question[:150]}")
    md_lines.append(f"\n**选项**:")
    for k, v in sorted(options.items()):
        mark = " ✅标准正确" if k in std else ""
        md_lines.append(f"\n- **{k}**: {v[:120]}{mark}")
    md_lines.append(f"\n**涉及文档**: `{doc_ids}`")
    md_lines.append(f"\n**标准答案**: `{std}`")

    # ── Stage 1: Full retrieval (Top-50) ──
    domain_config = route_domain(domain)
    keywords = extract_keywords_from_question(question)
    candidates = stage1_retrieve(keyword_index, doc_ids, question, options)

    s1_per_doc = dict(Counter(c['doc_id'] for c in candidates))
    md_lines.append(f"\n### Stage 1 检索层 (Top-{len(candidates)})")
    md_lines.append(f"\n| 指标 | 值 |")
    md_lines.append(f"\n|---|---|")
    md_lines.append(f"\n| 关键词(前15) | {', '.join(keywords[:15])} |")
    md_lines.append(f"\n| 候选总数 | {len(candidates)} |")
    md_lines.append(f"\n| 文档分布 | {s1_per_doc} |")

    # Check per-doc balance
    missing_in_s1 = [d for d in doc_ids if d not in s1_per_doc]
    if missing_in_s1:
        md_lines.append(f"\n| ⚠️ 文档缺失 | `{missing_in_s1}` — 该文档在 Stage 1 零命中！ |")

    # ── Stage 2: Evidence selection ──
    before_pf = len(candidates)
    candidates = prefilter_candidates(candidates, domain)
    evidence, s2_usage = stage2_filter(client, candidates, question, options, doc_ids)
    s2_per_doc = dict(Counter(e['doc_id'] for e in evidence))

    md_lines.append(f"\n### Stage 2 精选层 ({len(evidence)}段)")

    # Doc collapse detection
    for did in doc_ids:
        count = s2_per_doc.get(did, 0)
        if count == 0:
            md_lines.append(f"\n🚨 **[CRITICAL: DOC COLLAPSE]** `{did}` 在 Evidence 中 **0 段**！该文档完全消失！")
        elif count <= 2:
            md_lines.append(f"\n⚠️ **[WARNING: LOW COVERAGE]** `{did}` 仅有 **{count} 段**。")

    md_lines.append(f"\n| 指标 | 值 |")
    md_lines.append(f"\n|---|---|")
    md_lines.append(f"\n| Evidence 总数 | {len(evidence)} |")
    md_lines.append(f"\n| 文档分布 | {s2_per_doc} |")
    md_lines.append(f"\n| Stage 2 Token | in={s2_usage['input_tokens']} out={s2_usage['output_tokens']} |")

    # Evidence preview (first 3 per doc)
    md_lines.append(f"\n**证据预览**:")
    for e in evidence[:8]:
        snippet = e['text'][:150].replace('\n', ' ').replace('|', '/')
        md_lines.append(f"\n- `{e['doc_id'][:30]}` ({len(e['text'])}字): {snippet}...")

    # ── Expand ──
    evidence = expand_evidence(evidence, keyword_index, expand_by=1, domain=domain)
    if domain == 'insurance':
        evidence = allocate_per_doc(evidence, per_doc_cap=domain_config['per_doc_cap'],
                                    max_docs=domain_config['max_docs'])

    # ── Stage 3: Reasoning ──
    prompt_name = select_reasoning_prompt(domain, question)
    full_prompt = build_reasoning_prompt(prompt_name, evidence, question, options)
    result = reason(client, evidence, question, options, prompt_name, af)
    answer = normalize_answer(result.get('answer', ''), af)

    match = '✅' if answer == std else '❌'
    if answer == std:
        correct_count += 1

    md_lines.append(f"\n### Stage 3 推理层")
    md_lines.append(f"\n| 指标 | 值 |")
    md_lines.append(f"\n|---|---|")
    md_lines.append(f"\n| Prompt 模板 | `{prompt_name}` |")
    md_lines.append(f"\n| Prompt 长度 | {len(full_prompt)} 字符 |")
    md_lines.append(f"\n| 模型输出答案 | `{answer}` |")
    md_lines.append(f"\n| 标准答案 | `{std}` |")
    md_lines.append(f"\n| 匹配 | {match} |")
    md_lines.append(f"\n| Token | in={result.get('input_tokens',0)} out={result.get('output_tokens',0)} |")

    # Per-option reasoning
    md_lines.append(f"\n**逐选项推理**:")
    for r in result.get('results', []):
        opt = r.get('option','?')
        judgment = r.get('judgment','?')
        conf = r.get('confidence', 0)
        ev = r.get('evidence', {})
        reasoning = ev.get('reasoning', '')[:400]
        doc = ev.get('doc_id','')[:30]
        in_std = '✅' if opt in std else ''
        selected = '🔵已选' if opt in answer else '○未选'
        md_lines.append(f"\n#### {opt} — {judgment} (conf={conf}) {in_std} {selected}")
        md_lines.append(f"\n> 来源: `{doc}`")
        md_lines.append(f"\n> {reasoning}")

        # Check for key numbers
        opt_text = options.get(opt, '')
        numbers_in_opt = re.findall(r'\d+\.?\d*[万亿千百%％]*', opt_text)
        if numbers_in_opt:
            numbers_found_in_reasoning = [n for n in numbers_in_opt if n in reasoning]
            if numbers_found_in_reasoning:
                md_lines.append(f"\n> 🔢 推理中出现的数字: {numbers_found_in_reasoning}")

    # Missed options analysis
    missed = set(std) - set(answer)
    if missed:
        md_lines.append(f"\n**漏选分析**: 漏掉 `{missed}`")
        for opt in missed:
            opt_text = options[opt]
            nums = re.findall(r'\d+\.?\d*[万亿千百%％]*', opt_text)
            md_lines.append(f"\n- **{opt}**: {opt_text[:100]}")
            md_lines.append(f"\n  - 选项中的数字: {nums}")
            # Check if these numbers appear in evidence
            ev_text_all = ' '.join(e['text'] for e in evidence)
            found_in_ev = [n for n in nums if n in ev_text_all]
            if found_in_ev:
                md_lines.append(f"\n  - 证据中找到了: {found_in_ev} → 🧠 **模型推理失败**")
            else:
                md_lines.append(f"\n  - 证据中未找到 → 📡 **检索失败**")

md_lines.append(f"\n---\n## 📊 汇总\n")
md_lines.append(f"\n| 指标 | 值 |")
md_lines.append(f"\n|---|---|")
md_lines.append(f"\n| 正确 | {correct_count}/8 |")
md_lines.append(f"\n| 检索失败占比 | (见各题漏选分析) |")
md_lines.append(f"\n| 文档塌陷次数 | (见各题 CRITICAL WARNING) |")

# ── Optimization 4: Debug dump of raw markdown around key terms for failed questions ──
debug_dump = []
for qid in WRONG_QIDS:
    q = all_qs[qid]
    failed = STD[qid] != [t['model_answer'] for t in [{'model_answer':''}]][0]  # check if wrong
    # Always dump for wrong; check within the existing trace logic below
    # We'll dump after the loop

# Actually dump during the loop (add to the existing per-QID processing)
# Re-run the per-question analysis after the main loop
    # Check if this question had a retrieval failure
    for doc_id in q.get('doc_ids', []):
        md_path = os.path.join(PROCESSED_DIR, f'{doc_id}.md')
        if not os.path.exists(md_path):
            continue
        with open(md_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Search for key terms from missed options
        missed = set(STD[qid]) - set([t for t in [{}]] if False else [])  # placeholder
        # Just dump the first 3000 chars containing any option-related term
        opt_terms = re.findall(r'[一-鿿]{2,}', ' '.join(q['options'].values()))
        for term in opt_terms[:5]:
            idx = raw.find(term)
            if idx >= 0:
                start = max(0, idx - 1500)
                end = min(len(raw), idx + 1500)
                debug_dump.append(f"\n{'='*60}\nQID: {qid} | Doc: {doc_id} | Term: '{term}' (at char {idx})\n{'='*60}")
                debug_dump.append(raw[start:end])
                debug_dump.append(f"\n... (showing chars {start}-{end} of {len(raw)})\n")
                break  # one dump per doc per question

with open('output/debug_raw_md.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(debug_dump))

report = '\n'.join(md_lines)
with open('output/diagnostic_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"Report: output/diagnostic_report.md")
print(f"Raw dump: output/debug_raw_md.txt")
print(f"Correct: {correct_count}/8")
