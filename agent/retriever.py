"""Retriever: Stage 1 keyword-based + Stage 2 Qwen fine-grained filtering."""
import os
import re
import jieba
from collections import OrderedDict

# Load finance/legal custom dictionary to prevent compound terms from being fragmented
_dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finance_dict.txt")
if os.path.exists(_dict_path):
    jieba.load_userdict(_dict_path)
from agent.qwen_client import QwenClient, build_chat_message

# Chinese stop words to filter out from keywords
_STOP_WORDS = set("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 他 她 它 们 那 些 什么 而 为 所 以 之 与 及 或 但 被 从 把 对 将 能 可以 可能 需要 已经 还 又 再 更 最 请 关于 根据 按照 下列 以下 以上 其中 包括 下列 各项 是否 目前 相关 进行 以及".split())

# Meta-words from question boilerplate — pollute keyword extraction with "第一""第二""文档" etc.
_META_WORDS = {'第一', '第二', '第三', '第四', '两份', '一份', '文档', '材料', '选项', '说法', '符合', '正确', '哪些', '有关', '结合', '内容'}

# Minimum character length for Chinese keywords (filter out single chars like "条", "款")
_MIN_KEYWORD_LEN = 2


def extract_keywords_from_question(question: str) -> list[str]:
    keywords = []

    # 1. Extract quoted names from 《...》
    quoted = re.findall(r'《([^》]+)》', question)
    keywords.extend(quoted)

    # 2. Use entity patterns from indexer (company names, metrics, dates)
    from agent.indexer import ENTITY_PATTERNS
    for pattern_name in ["company", "metric", "date_threshold"]:
        pattern = ENTITY_PATTERNS.get(pattern_name)
        if pattern:
            for match in pattern.findall(question):
                keywords.append(match)

    # 3. Jieba Chinese word segmentation
    # Remove punctuation for cleaner segmentation
    clean_question = re.sub(r'[，。、；：？！\s]+', ' ', question)
    words = jieba.cut(clean_question)
    for w in words:
        w = w.strip()
        if len(w) >= _MIN_KEYWORD_LEN and w not in _STOP_WORDS:
            keywords.append(w)

    # 4. Also extract 2-4 char n-grams as fallback for financial terms
    # that jieba might not know (e.g., clause numbers, product codes)
    alpha_chars = re.findall(r'[一-鿿]{2,4}', question)
    for term in alpha_chars:
        if term not in _STOP_WORDS:
            keywords.append(term)

    # Deduplicate preserving order, filter meta-words
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen and kw not in _META_WORDS and kw not in _STOP_WORDS:
            seen.add(kw)
            unique.append(kw)
    return unique


def truncate_paragraphs(paragraphs: list[dict], max_tokens: int = 500) -> list[dict]:
    result = []
    for p in paragraphs:
        text = p["text"]
        if len(text) > max_tokens:
            text = text[:max_tokens] + "..."
        result.append({**p, "text": text})
    return result


def _clean_for_keywords(text: str) -> str:
    """Strip markdown symbols to avoid jieba noise from |, #, *, - etc."""
    import re
    return re.sub(r'[|#*>\-]+', ' ', text)


def stage1_retrieve_per_option(index: dict, doc_ids: list[str], question: str,
                               options: dict) -> list[dict]:
    """Per-option independent retrieval + union.

    Retrieves candidates for each option independently (question+option text),
    then unions results with best-score deduplication. Prevents keywords from
    different options from diluting each other's retrieval signals.
    """
    union_candidates = {}
    for _opt_label, opt_text in sorted(options.items()):
        opt_query = f"{question} {opt_text}"
        opt_results = _stage1_retrieve_single(index, doc_ids, opt_query)
        for r in opt_results:
            key = (r["doc_id"], r["para_id"])
            score = r.get("_score", 0)
            if key not in union_candidates or score > union_candidates[key].get("_score", 0):
                union_candidates[key] = r

    results = list(union_candidates.values())
    results.sort(key=lambda r: r.get("_score", 0), reverse=True)
    return results[:30]


def stage1_retrieve(index: dict, doc_ids: list[str], question: str,
                    options: dict | None = None) -> list[dict]:
    """Retrieve candidates.

    When options are provided, uses per-option independent retrieval + union.
    Otherwise falls back to question-only retrieval.
    """
    if options and len(options) >= 2:
        return stage1_retrieve_per_option(index, doc_ids, question, options)
    return _stage1_retrieve_single(index, doc_ids, question)


def _stage1_retrieve_single(index: dict, doc_ids: list[str], query: str) -> list[dict]:
    """Core retrieval for a single query string."""
    keywords = extract_keywords_from_question(query)
    per_doc_quota = max(15, 30 // max(1, len(doc_ids)))

    # For financial domains, prepend option-level metrics as high-priority keywords
    _FINANCIAL_DOMAINS = {'financial_reports', 'financial_contracts', 'research'}
    is_financial = any(d in _FINANCIAL_DOMAINS for d in doc_ids)

    if is_financial and options:
        target_metrics = extract_target_metrics(question, options)
        # Put metrics first so they get searched before generic keywords
        metric_kws = [m for m in target_metrics if m not in keywords]
        keywords = metric_kws + keywords

    all_results = []
    seen = set()
    doc_results_list = []
    for doc_id in doc_ids:
        if doc_id not in index:
            continue
        doc_results = []
        # Tokenize search_text ONCE per entry for weighted overlap scoring
        _DOMAIN_STOP_WORDS = {'应当', '的', '在', '内', '定', '关于', '机构', '或者', '进行',
                              '规定', '相关', '以下', '以上', '包括', '要求', '根据', '按照',
                              '可以', '需要', '已经', '所有', '必须', '及其', '经过', '通过'}
        entry_tokens = {}
        for entry in index[doc_id]:
            haystack = entry.get("search_text", entry["text"]).lower()
            entry_tokens[entry["para_id"]] = set(jieba.lcut(haystack))

        kw_token_sets = {}
        for kw in keywords:
            kw_token_sets[kw] = set(jieba.lcut(kw.lower())) - _DOMAIN_STOP_WORDS

        for kw, kw_tokens in kw_token_sets.items():
            if not kw_tokens:
                continue
            for entry in index[doc_id]:
                para_id = entry["para_id"]
                tokens = entry_tokens.get(para_id, set())
                if not tokens:
                    continue
                hits = kw_tokens & tokens
                if not hits:
                    continue

                # Separate semantic vs numeric hits
                semantic_hits = {t for t in hits if not (t.isdigit() or any(c in t for c in '日月年'))}
                numeric_hits = hits - semantic_hits

                # Numeric bonus ONLY if paragraph also has semantic keyword co-occurrence
                base = len(semantic_hits)
                num_bonus = len(numeric_hits) * 2 if semantic_hits else 0
                score = base + num_bonus

                # Synonym bonus: financial synonyms (金额↔规模↔额度, 评级↔级别↔等级)
                _SYNONYM_MAP = {
                    '金额': {'金额', '规模', '总额', '额度', '限额'},
                    '规模': {'金额', '规模', '总额', '额度', '限额'},
                    '评级': {'评级', '级别', '等级', '信用等级'},
                }
                for t in hits:
                    if t in _SYNONYM_MAP:
                        for syn in _SYNONYM_MAP[t]:
                            if syn in tokens:
                                score += 1
                key = (doc_id, para_id)
                if key not in seen:
                    seen.add(key)
                    doc_results.append({
                        "doc_id": doc_id,
                        "para_id": para_id,
                        "text": entry["text"],
                        "_score": score,
                    })
                else:
                    # Update max score for the same paragraph
                    for dr in doc_results:
                        if dr["doc_id"] == doc_id and dr["para_id"] == para_id:
                            dr["_score"] = max(dr.get("_score", 0), score)
                            break
            # Sort by accumulated score before truncation
            doc_results.sort(key=lambda r: r.get("_score", 0), reverse=True)
            if len(doc_results) > per_doc_quota * 3:
                doc_results = doc_results[:per_doc_quota * 3]
        doc_results_list.append(doc_results)

    # Score and re-rank: prefer paragraphs matching multiple target metrics
    try:
        target_metrics = extract_target_metrics(question, options) if options else _METRIC_NAMES
    except Exception:
        target_metrics = _METRIC_NAMES
    for doc_results in doc_results_list:
        for r in doc_results:
            haystack = r.get("search_text", r["text"])
            score = sum(1 for m in target_metrics if m in haystack)
            r["_score"] = score
        doc_results.sort(key=lambda r: r.get("_score", 0), reverse=True)
        # Keep only top per_doc_quota
        doc_results[:] = doc_results[:per_doc_quota]

    # Per-doc minimum guarantee: at least min_per_doc from each document
    min_per_doc = max(5, 30 // max(1, len(doc_ids) * 2))
    guaranteed = []
    for doc_results in doc_results_list:
        guaranteed.extend(doc_results[:min_per_doc])

    # Interleave remaining results for balanced coverage
    remaining_slots = 30 - len(guaranteed)
    max_len = max((len(d) for d in doc_results_list), default=0)
    interleaved = []
    seen_keys = {(r['doc_id'], r['para_id']) for r in guaranteed}
    for i in range(min_per_doc, max_len):
        for doc_results in doc_results_list:
            if i < len(doc_results) and len(interleaved) < remaining_slots:
                r = doc_results[i]
                key = (r['doc_id'], r['para_id'])
                if key not in seen_keys:
                    seen_keys.add(key)
                    interleaved.append(r)

    all_results = guaranteed + interleaved
    return all_results[:30]


def allocate_per_doc(evidence: list[dict], per_doc_cap: int,
                     max_docs: int) -> list[dict]:
    doc_groups = OrderedDict()
    for e in evidence:
        doc_id = e["doc_id"]
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append(e)

    result = []
    for doc_id, entries in list(doc_groups.items())[:max_docs]:
        doc_chars = 0
        for entry in entries:
            text = entry["text"]
            if doc_chars + len(text) > per_doc_cap:
                remaining = per_doc_cap - doc_chars
                if remaining > 50:
                    entry = {**entry, "text": text[:remaining] + "..."}
                    result.append(entry)
                break
            result.append(entry)
            doc_chars += len(text)
    return result


# Financial metric names for targeted retrieval
_METRIC_NAMES = [
    '营业收入', '净利润', '归属于母公司', '归属于上市公司股东',
    '研发投入', '现金流量净额', '经营活动.*?现金流',
    '资产负债率', '每股收益', '基本每股收益', '稀释每股收益',
    '毛利率', '营业成本', '总资产', '净资产', '净资产收益率',
    '分红', '派息', '现金分红', '利润分配',
    '保费', '现金价值', '身故保险金', '退保',
    '募集资金', '发行金额', '发行规模', '主体信用评级',
    # Convertible bond specific terms
    '转股价格', '向下修正', '赎回条款', '有条件赎回', '到期赎回',
    '回售条款', '股票代码', '发行日期', '初始转股', '票面利率',
    '身故', '保险金', '领取', '账户价值',
]


def extract_target_metrics(question: str, options: dict) -> list[str]:
    """Extract specific financial metric names from question + options."""
    text = question + ' ' + ' '.join(options.values())
    found = []
    for m in _METRIC_NAMES:
        if re.search(m, text):
            found.append(m)
    return found


# Noise patterns for pre-filtering irrelevant paragraphs
_NOISE_PATTERNS = [
    re.compile(r'^第[一二三四五六七八九十百千\d]+章'),  # Chapter titles
    re.compile(r'董事长致辞|总裁致辞|致股东|致各位股东'),  # Chairman letters
    re.compile(r'目\s*录|CONTENTS|目  录'),  # TOC
    re.compile(r'^\s*\d+\s*$'),  # Pure page numbers
    re.compile(r'声明|免责|风险提示|重要提示'),  # Disclaimers
    re.compile(r'释义项|释义内容'),  # Definition tables
    re.compile(r'备查文件|备查'),  # Reference file lists
]

# Domains that benefit from noise filtering
_NOISE_FILTER_DOMAINS = {'financial_reports', 'financial_contracts', 'research'}


def prefilter_candidates(candidates: list[dict], domain: str) -> list[dict]:
    """Filter out structural noise paragraphs before Stage 2."""
    if domain not in _NOISE_FILTER_DOMAINS:
        return candidates

    filtered = []
    for c in candidates:
        text = c['text']
        if not any(p.search(text) for p in _NOISE_PATTERNS):
            filtered.append(c)

    return filtered if len(filtered) >= 10 else candidates


def _ensure_doc_coverage(evidence: list[dict], candidates: list[dict],
                         expected_doc_ids: list[str], max_total: int = 5) -> list[dict]:
    """Ensure at least one paragraph from each expected document is in evidence."""
    if len(expected_doc_ids) <= 1:
        return evidence[:max_total]

    covered = {e['doc_id'] for e in evidence}
    missing = [d for d in expected_doc_ids if d not in covered]

    result = list(evidence)
    for doc_id in missing:
        # Find the best candidate from the missing document
        for c in candidates:
            if c['doc_id'] == doc_id and c not in result:
                result.append(c)
                break

    return result[:max_total]


def expand_evidence(evidence: list[dict], keyword_index: dict,
                    expand_by: int = 1, domain: str | None = None) -> list[dict]:
    """Bi-directional context expansion.

    For each evidence paragraph, includes ±N adjacent paragraphs.
    If the paragraph starts with a subordinate marker ((一), 1., a)),
    also pulls the preceding paragraph (the main clause).
    """
    import re
    expanded = list(evidence)
    seen = {(e['doc_id'], e['para_id']) for e in evidence}

    # ── Optimization 2: Table header stitching ──
    # For table chunks, trace back to find the header row (|---|) and prepend it
    for e in evidence:
        text = e['text']
        if '|---' in text or text.strip().startswith('|'):
            entries = keyword_index.get(e['doc_id'], [])
            # Search backward for the header row
            for back in range(1, 6):
                target_id = e['para_id'] - back
                if target_id < 0:
                    break
                for entry in entries:
                    if entry['para_id'] == target_id:
                        prev_text = entry['text']
                        if '|---' in prev_text or (prev_text.strip().startswith('|') and '---' in prev_text):
                            # Found header — inject it above current chunk
                            header_lines = []
                            for line in prev_text.split('\n'):
                                if line.strip().startswith('|'):
                                    header_lines.append(line)
                            if header_lines and (e['doc_id'], target_id) not in seen:
                                expanded.append({
                                    'doc_id': e['doc_id'],
                                    'para_id': target_id,
                                    'text': '\n'.join(header_lines),
                                })
                                seen.add((e['doc_id'], target_id))
                            break
                else:
                    continue
                break
        doc_id = e['doc_id']
        para_id = e['para_id']
        entries = keyword_index.get(doc_id, [])

        # Detect subordinate clause markers: needs preceding context
        # \s* after the marker, not \s — "1.应当" has no space after "1."
        is_subordinate = bool(re.match(r'^\s*[\(（]\s*[一二三四五六七八九十\d]+\s*[\)）]', text)
                              or re.match(r'^\s*\d+[.)]', text)
                              or re.match(r'^\s*[a-z][.)]', text, re.IGNORECASE))

        offsets_to_try = []
        # Instruction 2: res/fc domains expand ±1 for all evidence
        if domain in ('research', 'financial_contracts'):
            offsets_to_try = [-1, 1]
        elif is_subordinate:
            offsets_to_try = [-1, 1]  # Pull both preceding main clause + next
        else:
            offsets_to_try = [1]  # Only pull next

        for offset in offsets_to_try:
            if offset == 0:
                continue
            target_id = para_id + offset
            if target_id < 0 or (doc_id, target_id) in seen:
                continue
            for entry in entries:
                if entry['para_id'] == target_id:
                    expanded.append({
                        'doc_id': doc_id,
                        'para_id': target_id,
                        'text': entry['text'],
                    })
                    seen.add((doc_id, target_id))
                    break

    return expanded


def stage2_filter(client: QwenClient, candidates: list[dict],
                  question: str, options: dict,
                  expected_doc_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    if len(candidates) <= 5:
        return candidates, {"input_tokens": 0, "output_tokens": 0}

    # Pre-filter noise
    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "stage2_filter.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    metrics = extract_target_metrics(question, options)
    metrics_text = "、".join(metrics) if metrics else "无特定指标要求"

    options_text = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    # Dynamic truncation: markdown tables get 4000 chars, others 2000
    def _truncate_candidate(text: str) -> str:
        is_table = "|---" in text or text.strip().startswith("|")
        limit = 4000 if is_table else 2000
        return text[:limit] if len(text) > limit else text

    candidates_text = "\n\n".join(
        f"#{i}: [{c['doc_id']}] {_truncate_candidate(c['text'])}"
        for i, c in enumerate(candidates)
    )

    prompt = template.format(
        question=question,
        options=options_text,
        candidates=candidates_text,
        metrics=metrics_text,
    )

    response = client.chat(
        messages=[build_chat_message("user", prompt)],
        temperature=0,
        max_tokens=500,
    )

    content = response["content"].strip()
    numbers = re.findall(r'\d+', content)
    ranked_ids = [int(n) for n in numbers if int(n) < len(candidates)]

    # ── Instruction 1 (upgraded): Option-driven targeted feature matching ──
    # Extract numbers + core noun phrases from options, filter generic boilerplate
    _GENERIC_WORDS = {'文档', '低于', '高于', '描述', '符合', '正确', '第一', '第二',
                      '两份', '一份', '材料', '以上', '以下', '包括', '是否', '相关'}
    _option_features = set()
    _opt_text = " ".join(options.values())
    for num in re.findall(r'\d+\.?\d*[万亿千百%％]*', _opt_text):
        _option_features.add(num)
    for entity in re.findall(r'[一-鿿]{2,6}(?:证券|银行|保险|基金|股份|集团|管理办法|指引|准则)', _opt_text):
        _option_features.add(entity)
    # Extract core noun phrases (2-4 chars, not generic)
    phrases = re.findall(r'[一-鿿]{2,4}', _opt_text)
    for p in phrases:
        if p not in _GENERIC_WORDS and p not in ('的','了','在','是','有','和','就','不','人','都','一'):
            _option_features.add(p)

    # ── Numeric density filter ──
    # Detect if any option has quantitative intent (金额, 规模, 上限, 比例, etc.)
    _NUMERIC_INTENT_WORDS = {'金额', '规模', '上限', '下限', '比例', '价格',
                              '利率', '费用', '低于', '超过', '占.*比例', '增速',
                              # Optimization 6: time/regulatory terms
                              '工作日', '天内', '个月', '年内', '届满', '之日起',
                              '年满', '满.*年', '日以上', '年以下', '年以', '日起'}
    _has_numeric_intent = any(
        re.search(w, _opt_text) for w in _NUMERIC_INTENT_WORDS
    )

    # Scan candidates for golden hints
    _golden_hints = []
    for i, c in enumerate(candidates):
        search_field = c.get("search_text", c["text"])
        for feat in _option_features:
            if feat in search_field:
                # Per-feature numeric gate: only require numbers if THIS feature is numeric
                feat_is_numeric = any(re.search(w, feat) for w in _NUMERIC_INTENT_WORDS)
                if feat_is_numeric and not re.search(r'\d+', search_field):
                    continue  # e.g. "金额" matched but paragraph has no numbers → skip
                _golden_hints.append(c)
                break  # one match is enough

    # Dynamic top-N: more docs = more evidence needed
    max_docs = len(expected_doc_ids) if expected_doc_ids else 1
    top_n = 15 if max_docs >= 3 else (12 if max_docs >= 2 else 8)

    top = []
    seen = set()
    for idx in ranked_ids[:top_n]:
        if idx not in seen:
            seen.add(idx)
            top.append(candidates[idx])

    # Ensure each expected document is covered
    if expected_doc_ids and len(expected_doc_ids) > 1:
        top = _ensure_doc_coverage(top, candidates, expected_doc_ids, max_total=top_n + 3)

    token_usage = {
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
    }
    # ── Instruction 1 (continued): Force golden hints into evidence ──
    # Golden hints get guaranteed inclusion, immune to filtering
    # Cap at 5 to prevent token overflow
    _seen_hint_texts = {e['text'][:100] for e in top}
    _hints_added = 0
    for gh in _golden_hints:
        if gh['text'][:100] not in _seen_hint_texts and _hints_added < 5:
            top.insert(0, gh)  # prepend for maximum LLM attention
            _seen_hint_texts.add(gh['text'][:100])
            _hints_added += 1

    # ── Optimization 5: Per-document Top-3 minimum ──
    # For multi-doc questions, if any doc has <3 in evidence, inject its best unscored
    if expected_doc_ids and len(expected_doc_ids) >= 2:
        for did in expected_doc_ids:
            already_in_top = [c for c in top if c.get("doc_id") == did]
            if len(already_in_top) < 3:
                scored_candidates = sorted(
                    [c for c in candidates if c.get("doc_id") == did and c not in top],
                    key=lambda c: c.get("_score", 0), reverse=True
                )
                need = 3 - len(already_in_top)
                for dc in scored_candidates[:need]:
                    top.append(dc)

    # Document-balanced rebalancing: allocate equal slots per document
    max_docs = len(expected_doc_ids) if expected_doc_ids else 1
    if max_docs >= 2 and len(top) > max_docs:
        per_doc = max(1, len(top) // max_docs)
        balanced = []
        doc_slots = {}
        for c in top:
            did = c.get("doc_id", "")
            if doc_slots.get(did, 0) < per_doc:
                doc_slots[did] = doc_slots.get(did, 0) + 1
                balanced.append(c)
        # Fill remaining slots with best unscored from underrepresented docs
        for c in candidates:
            did = c.get("doc_id", "")
            if doc_slots.get(did, 0) < per_doc and len(balanced) < len(top):
                doc_slots[did] = doc_slots.get(did, 0) + 1
                balanced.append(c)
        if len(balanced) >= max_docs * 2:
            top = balanced

    return top, token_usage


def should_skip_stage2(candidates: list[dict], question: str) -> bool:
    if len(candidates) <= 5:
        return True
    from agent.indexer import ARTICLE_NUM_PATTERN
    clause_nums = ARTICLE_NUM_PATTERN.findall(question)
    if clause_nums:
        top3_texts = " ".join(c["text"] for c in candidates[:3])
        for cn in clause_nums:
            if cn in top3_texts:
                return True
    return False
