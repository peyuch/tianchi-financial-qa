# Financial Long-Document QA Agent — Design Spec

> **Competition**: TianChi Financial Long-Text QA — A榜 (100 questions) + B榜 (100 questions)
> **Constraint**: Reasoning stage MUST use Qwen-series models (百炼/魔搭); preprocessing can use any tools but non-Qwen semantic results must not leak into retrieval/reasoning.
> **Scoring**: `FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)`, TokenBudget = 5,000,000

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  PREPROCESSING (MinerU + HTML path, offline)    │
│  PDF → MinerU → markdown (headings, page numbers, tables,      │
│                            article/clause numbers preserved)    │
│  HTML (regulatory domain, 377 files) → BeautifulSoup/lxml      │
│    → extract plain text → markdown (preserve heading tags,     │
│      article numbers, paragraphs)                              │
│  TXT (regulatory domain, 6 files) → read directly, segment     │
│    by blank lines / article markers                            │
│  Per-doc output: {doc_id}.md in uniform format                 │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  INDEXING (zero-token, offline)                │
│  Layer 1: Keyword inverted index                              │
│    · Clause/article numbers (第四十七条, Article 3.2)          │
│    · Entity names (company, product, regulation)              │
│    · Metric names (营业收入, 研发投入占比, 现金价值)            │
│    · Dates and thresholds (施行日期, 资产负债率超过70%)         │
│  Layer 2: Chapter tree per document (heading hierarchy)        │
│  Layer 3: [B榜 plug] Qwen Embedding vector index (API-native)  │
│    · Stub for A榜, activated for B榜 blind retrieval           │
│                                                               │
│  Per-doc summary (zero-token, extracted from document          │
│  structure during indexing, stored in unified format):         │
│  {                                                             │
│    "doc_id": "annual_byd_2024_report",                         │
│    "domain": "financial_reports",                              │
│    "summary_keywords": ["比亚迪", "2024年报", "营业收入",       │
│                         "净利润", "研发投入"],                   │
│    "toc": ["第一节 重要提示", "第二节 公司简介", ...],           │
│    "meta": {"year": "2024", "company": "BYD", "pages": 290}    │
│  }                                                             │
│  · Keywords extracted from: TOC headings + first-section       │
│    product/company/regulation names + metric terms             │
│  · Used by B榜 for document-level coarse filtering before      │
│    paragraph retrieval. Avoids full Qwen-generated summaries   │
│    (which would cost ~1M+ tokens for the corpus).              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  DOMAIN ROUTER (rule-based, zero-token)        │
│  Reads question JSON → routes by `domain` field:               │
│                                                               │
│  financial_reports → numerical_comparison path                 │
│    context_limit: 5000 token, needs multi-table extraction     │
│  regulatory → rule_comprehension path                         │
│    context_limit: 2000 token, clause-focused search            │
│  insurance → cross_policy_comparison path                     │
│    context_limit: 3000 token, avg 3.55 docs/question          │
│  financial_contracts → clause_verification path               │
│    context_limit: 4000 token, bond terms + financial data      │
│  research → data_verification path                            │
│    context_limit: 3000 token, numerical + temporal checks      │
│                                                               │
│  Also detects: answer_format (mcq/multi/tf) → output rules     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  TWO-STAGE RETRIEVAL                           │
│                                                               │
│  Stage 1: Rule-based coarse retrieval (zero-token)             │
│    · Clause number exact match → direct hit                   │
│    · Entity/company name keyword match in inverted index      │
│    · Metric name grep across indexed sections                 │
│    · Recall: Top-20~30 candidate paragraphs                   │
│    · Each paragraph truncated to ≤500 tokens                   │
│                                                               │
│  Stage 2: Qwen fine-grained filtering (1 API call, gated)     │
│    · Gate check: if Stage 1 results are from direct clause     │
│      match AND top-3 relevance is clear → skip Stage 2        │
│    · Prompt: "以下段落中，哪些包含能判断各选项正误的具体       │
│      数据或规则？按相关性从高到低排序，输出段落编号列表。"      │
│    · Ranked list output is more parseable than yes/no per item │
│    · Input: Top-20~30 candidates (≤500 token each)             │
│             + question + all options ≈ 10,000 token            │
│    · Output: Top-5 evidence paragraph IDs, sorted by relevance │
│                                                               │
│  B榜 mode: Stage 1 keyword fails → Qwen Embedding vector      │
│    search across all domain documents → then Stage 2           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  BATCH REASONING (1 API call)                  │
│                                                               │
│  Input: Top-5 evidence paragraphs + question + ALL 4 options   │
│  Prompt structure:                                            │
│    1. Evidence context (domain-specific token cap applied)     │
│    2. Question + all options A/B/C/D at once                  │
│    3. Chain-of-thought: judge each option against evidence     │
│    4. Numerical questions MUST show computation chain:          │
│       "原始数据来源 → 公式 → 逐步计算 → 结果"                   │
│       禁止直接给出最终数字，必须展示中间步骤。                   │
│    5. Structured output with confidence score per option       │
│                                                               │
│  Output format (embedded in prompt):                           │
│  {                                                            │
│    "results": [                                               │
│      {"option": "A", "judgment": "正确",                       │
│       "evidence": {"doc_id": "...", "section": "...",         │
│                    "quote": "..."}, "confidence": 0.85},      │
│      ...                                                      │
│    ],                                                         │
│    "answer": "AC"                                             │
│  }                                                            │
│                                                               │
│  Prompt ends with hard constraint:                             │
│  "最后严格按以上JSON格式输出，不要输出任何其他内容。"             │
│                                                               │
│  Parser includes fallback: if JSON.parse fails, regex-extract │
│  option letters and judgment strings from raw output.         │
│  At minimum, salvage the answer string to avoid scoring zero.  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  ANSWER VALIDATION + FALLBACK                  │
│                                                               │
│  High confidence (all options ≥0.8) → emit answer directly    │
│  Low confidence (any option <0.5) → supplementary retrieval   │
│    · Query Qwen: "what evidence is missing?"                  │
│    · One additional retrieval round (max 2 rounds total)      │
│    · Re-run reasoning with augmented context                  │
│                                                               │
│  Format normalization:                                        │
│    · mcq/tf: take first valid letter, uppercase               │
│    · multi: sort letters, deduplicate, no separators (e.g. AC)│
│    · Invalid output → mark as empty (auto-wrong per rules)    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  OUTPUT: answer.csv + evidence.json            │
└──────────────────────────────────────────────────────────────┘
```

## Two-Layer Zero-Token Routing

Layer 1 — `domain` field (always present in question JSON):
- Sets: retrieval strategy, document scope, context limits
- Zero ambiguity: domain is guaranteed

Layer 2 — keyword rules within domain (selects reasoning prompt template):
- `financial_reports` + `同比|环比|增长|变化|较上年|较20\d{2}年|增速|降幅|连续\d+年` → temporal comparison prompt
- `financial_reports` + `计算|合计|占比|金额` → numerical computation prompt
- `insurance` + `计算|赔付|退保|金额` → numerical computation prompt
- `insurance` + 默认 → clause comparison prompt
- `regulatory` + 默认 → regulation compliance prompt
- `financial_contracts` + 默认 → bond clause verification prompt
- `research` + `同比|增长|变化|趋势|历年` → temporal comparison prompt
- `research` + 默认 → data verification prompt
- Fallback for all: domain default prompt

Misclassification is safe — prompts only nudge reasoning direction; evidence paragraphs contain all necessary data regardless. Keywords are tuned for zero false-positives (conservative matching).

## Domain-Specific Configuration

| Parameter | financial_reports | regulatory | insurance | financial_contracts | research |
|---|---|---|---|---|---|
| Context token cap | 5000 total | 2000 total | **1000/doc × max 4 docs** | 4000 total | 3000 total |
| Per-doc allocation | 2500 × 2 docs | 1000 × 2 docs | 1000 × up to 4 docs | 2000 × 2 docs | 1500 × 2 docs |
| Avg docs/question | 2.0 | 2.0 | **3.55** | 2.0 | 2.0 |
| Numerical focus | 100% | 0% | 25% | 0% | 25% |
| Temporal focus | 55% | 0% | 0% | 10% | 50% |
| Stage 2 skip rate (est.) | Low (complex data) | High (clause direct hit) | Medium | Medium | Medium |
| Per-doc avg pages | 215 | ~10 (clauses) | 19 | 309 | 39 |

Insurance uses per-document allocation instead of total cap because questions average 3.55 docs. A 3000 total cap risks one document crowding out others. `1000 token × max 4 docs` ensures every referenced document contributes evidence.

## API Call Budget (per 100 questions)

Fallback structure: low-confidence → retriever adds evidence → 1 merged re-reasoning call (no separate diagnosis step).

### Three-Scenario Breakdown

| Component | Best Case | Target Case | Worst Case |
|---|---|---|---|
| **Stage 2 calls** | 10 calls | 50 calls | 100 calls |
| Stage 2 tokens/call | 10,000 | 10,000 | 12,000 (thick docs) |
| Stage 2 subtotal | **100,000** | **500,000** | **1,200,000** |
| **Batch reasoning calls** | 100 | 100 | 100 |
| Reasoning tokens/call | 18,000 | 20,000 | 22,000 |
| Reasoning subtotal | **1,800,000** | **2,000,000** | **2,200,000** |
| **Retry calls** | 3 | 15 | 30 |
| Retry tokens/call | 20,000 | 20,000 | 25,000 |
| Retry subtotal | **60,000** | **300,000** | **750,000** |
| **Total** | **~1,960,000** | **~2,800,000** | **~4,150,000** |

### Score Projection (at Accuracy = 70%)

| Scenario | Total Tokens | TokenScore | Coefficient | FinalScore |
|---|---|---|---|---|
| Best | 1.96M | 0.608 | 0.882 | **61.8** |
| Target | 2.80M | 0.440 | 0.832 | **58.2** |
| Worst | 4.15M | 0.170 | 0.751 | **52.6** |

## doc_id → File Mapping

All 5 domains use **direct mapping**: `doc_id + extension = filename`. No metadata file needed.

| Domain | Pattern | Example |
|---|---|---|
| insurance | `{doc_id}.pdf` | doc_id="1" → 1.pdf |
| financial_reports | `{doc_id}.PDF` | doc_id="annual_byd_2024_report" → annual_byd_2024_report.PDF |
| financial_contracts | `{doc_id}.pdf` | doc_id="text01" → text01.pdf |
| research | `{doc_id}.pdf` | doc_id="pack2_text01" → pack2_text01.pdf |
| regulatory | `{doc_id}.txt` (or `.html` for csrc_*) | doc_id → same filename in raw/regulatory/txt/ or raw/regulatory/html/ |

Regulatory files are split across three subdirectories (`html/`, `txt/`, `attachments/`). The doc_id is the exact filename without extension. Lookup: try `.txt` first, then `.html`, then check `attachments/` for `.pdf`.

## Key Design Decisions (with data evidence)

1. **Domain-based routing instead of LLM question classification**: The `domain` field predicts question characteristics with near-perfect reliability (e.g., financial_reports = 100% numerical; regulatory = 0% numerical). Saves 100 API calls.

2. **Batch reasoning (all 4 options in one call)**: Reduces reasoning tokens by ~60% vs per-option calls. Prompt includes all options alongside evidence context.

3. **Domain-specific context limits** (not one-size-fits-all): Validated against actual document page counts and question types per domain.

4. **Stage 2 "fast path" gating**: Questions with direct clause-number matches skip the Qwen filtering call. In insurance and regulatory domains especially, this saves ~50% of Stage 2 calls.

5. **No full-document summarization via Qwen**: Instead, use MinerU-extracted TOC/first-section as zero-token document summaries for B榜 document filtering.

6. **Local ReAct for low-confidence answers only**: Cap supplementary retrieval at 2 rounds. Most questions resolve in the standard 2-3 call pipeline.

## File Structure (target)

```
TianChi/
├── data/
│   ├── processed/          # MinerU output: {doc_id}.md per document
│   ├── indexes/
│   │   ├── keyword_index.json   # Inverted index
│   │   └── chapter_trees.json   # Per-doc heading hierarchy
│   └── summaries/          # TOC/first-section summaries (extracted, not generated)
├── agent/
│   ├── preprocessor.py     # MinerU (PDF) + BeautifulSoup (HTML) + TXT → unified .md
│   ├── indexer.py          # Build keyword + chapter indexes
│   ├── retriever.py        # Two-stage retrieval (rule + Qwen)
│   ├── reasoner.py         # Batch CoT reasoning
│   ├── validator.py        # Answer formatting + confidence check
│   └── pipeline.py         # Main orchestration
├── prompts/
│   ├── stage2_filter.txt   # "哪些段落包含能判断选项正误的具体数据/规则？按相关性排序输出编号"
│   ├── batch_reasoning.txt # Per-domain CoT + numerical step-by-step + JSON hard constraint
│   └── fallback.txt        # "当前证据不足以判断X选项，需要补充哪些信息？"
├── output/
│   ├── answer.csv
│   └── evidence.json
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-10-financial-qa-agent-design.md
```

## B榜 Preparation (design decisions, not yet implemented)

- Qwen Embedding vector index plug already architected in retriever module
- Document TOC summaries enable keyword-based document-level filtering when doc_ids are absent
- Same reasoning pipeline works unchanged — only retrieval Stage 1 expands from "load specified docs" to "search all domain docs"
- Target B榜 readiness: A榜 pipeline with retrieval module swapped
