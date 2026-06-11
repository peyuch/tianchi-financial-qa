# 金融长文档 QA Agent — 开发过程总结

## 项目概述

阿里云天池金融长文档问答比赛 A 榜。100 道选择题（单选/多选/判断），覆盖 5 个领域（保险条款、监管法规、金融合同、财报、研报），基于 68 份 PDF/HTML/TXT 文档作答。约束：推理阶段只能用 Qwen 系列模型 API；评分 = 准确率 + Token 效率加权。

---

## 一、架构设计（Brainstorming 阶段）

### 核心决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 编程语言 | Python | MinerU/百炼SDK生态 |
| PDF 解析 | PyMuPDF（MinerU 备选） | MinerU 太慢(311页需1h) |
| 模型 API | 百炼 DashScope `qwen-plus` | 稳定、速度快 |
| 架构模式 | 混合式 Pipeline | 管道主控 + 推理环节CoT |

### Pipeline 架构

```
PDF/HTML/TXT → Preprocessor → Indexer → DomainRouter → Retriever(Stage1+2) → Reasoner → Validator → Output
```

**7 个模块**：
1. **Preprocessor** — PyMuPDF( PDF) / BeautifulSoup(HTML) / TXT → 统一 .md
2. **Indexer** — 关键词倒排索引 + 章节树 + 文档摘要
3. **DomainRouter** — 零 Token 领域路由（domain 字段 + 关键词规则）
4. **Retriever** — Stage 1 关键词检索 + Stage 2 Qwen 精筛
5. **Reasoner** — 批量 CoT 推理（4 选项一次调用）
6. **Validator** — 答案标准化 + 置信度校验
7. **Pipeline** — 主流程编排 + 输出 answer.csv + evidence.json

---

## 二、技术栈

| 组件 | 技术 |
|---|---|
| PDF 提取 | PyMuPDF (fitz) |
| HTML 解析 | BeautifulSoup4 + lxml |
| 中文分词 | jieba |
| LLM API | DashScope SDK（百炼） |
| 模型 | `qwen-plus` (Qwen3.6 系列) |
| 测试框架 | pytest |
| 环境管理 | conda (knowledge-platform env, Python 3.11) |

---

## 三、执行过程中遇到的问题与解决方案

### 问题 1：MinerU 安装与性能

**难度：★★★☆☆（中等）**

MinerU 是赛题推荐的 PDF 解析工具，保留表格结构。但遇到三个障碍：
- 安装需要下载 ~12GB 模型文件（网络中断一次）
- 首次使用需下载多个模型到本地
- 实际运行速度极慢：311 页债券募集书需 **1 小时+**

**解决**：使用 PyMuPDF（fitz）作为轻量级替代，通过 `page.get_text("text")` 提取纯文本。后续开发了 `merge_orphan_number_lines` 函数来修复 PyMuPDF 产生的表格数据碎片化问题。MinerU 保留为长期优化项。

---

### 问题 2：DashScope API 模型名称混乱

**难度：★★★☆☆（中等）**

**现象**：赛题要求使用 Qwen3.6-plus，但百炼平台上 `qwen3.6-plus` 调用 `Generation.call` 返回 400 错误。尝试 `qwen3.7-plus` 同样失败。

**排查过程**：逐一测试所有模型 ID：

| 模型 ID | Generation.call | MultiModalConversation.call |
|---|---|---|
| `qwen-plus` | ✅ 200 | ❌ 400 |
| `qwen3.6-plus` | ❌ 400 | ✅ 200 |
| `qwen3.7-plus` | ❌ 400 | ✅ 200 |

**根因**：百炼有两套 API 命名空间。`qwen3.6-plus`/`qwen3.7-plus` 仅存在于多模态 API；纯文本 API 中 `qwen-plus` 才是 Qwen3.6 代模型。

**解决**：最终使用 `qwen-plus` 通过 `Generation.call` 调用，确认与赛题基准模型能力一致。

---

### 问题 3：DashScope SDK 响应格式变更

**难度：★★☆☆☆（简单）**

**现象**：`qwen_client.py` 中 `response.output.choices[0].message.content` 报 `NoneType` 错误，但 API 返回 `status_code=200`。

**排查**：打印 `response.output` 结构，发现新版 DashScope SDK 将 `output` 改为 dict 类型，内容在 `output["text"]` 字段而非 `output.choices[0].message.content`。

**解决**：编写 `_extract_content()` 和 `_extract_tokens()` 兼容两种格式。

---

### 问题 4：魔搭社区 API 超时

**难度：★★☆☆☆（简单）**

**现象**：切换魔搭社区免费 API 后，简单对话正常但推理调用超时（ReadTimeout）。

**根因**：魔搭免费 API 有服务端超时限制，大 prompt（证据段 + CoT 指令）处理时间超过限制。

**解决**：放弃魔搭，继续使用百炼 DashScope（付费但稳定）。

---

### 问题 5：关键词提取失效——空答案率 25%

**难度：★★★★★（困难，核心问题）**

这是整个开发过程中最关键的问题，经历了多轮诊断和修复。

**现象**：首次全量运行 100 题，25 题答案为空白。

**诊断链**：

#### 5a. 中文分词缺失

`extract_keywords_from_question` 使用 `[一-鿿]{3,}` 正则匹配 3+ 汉字，导致两个致命问题：
- 2 字词（"股东""章程""职权"）全部漏掉
- 整句被当成一个巨型"关键词"（如"结合上市公司治理准则与章程指引"）

**修复**：引入 jieba 分词，添加停用词过滤，保留 2-4 字 n-gram 作为 fallback。
**效果**：Stage 1 候选段从 0 恢复到 30。

#### 5b. 财务数据表格被段落切分破坏

年报 markdown 中表格数据被 `\n\n` 切分导致标签和数值分离：
```
段落A: "营业收入（千元）"        ← 只有标签
段落B: "362,012,554"              ← 只有数字，在另一段！
```
关键词"营业收入"命中段落A，但段落A没有数字。模型看到"营业收入（千元）"无法计算，诚实输出"无法判断"。

**修复**：`merge_orphan_number_lines()` — 识别"上一行中文标签 + 下一行纯数字"模式，合并为一行。
**效果**：空答案从 25→15。

#### 5c. 第一份文档独占检索配额

`stage1_retrieve` 中 `if len(results) >= 30: break` 在第一个文档匹配满 30 条后直接跳出，第二个文档完全没被搜索。例如 `fin_a_009`（比亚迪 vs 中国移动），比亚迪年报占满 30 个名额，中国移动零覆盖。

**修复**：改为按文档交替排列 + 每文档配额制。
**效果**：跨文档题从零覆盖变为 15+15 均衡。

#### 5d. 关键词"比亚迪"匹配全篇 290 段

年报每页页眉都含公司名，一个关键词就触发 290 个段落匹配，占满配额。这些匹配包含大量封面页、目录、董事会名单等无关段落，真正的财务数据段被挤出 Top-15。

**修复**：财务指标关键词（"营业收入""净利润"等）优先搜索，实体词（"比亚迪""中国移动"）作为兜底。
**效果**：Stage 1 候选质量大幅提升。

#### 5e. 多指标评分排序

**修复**：对候选段按命中指标数打分排序——同时匹配"营业收入""净利润""研发投入"的财务数据表段排最前面，只匹配一个词的封面段排后面。
**效果**：财务数据段进入 Top-5。

#### 5f. 证据扩展 + 段落截断

- Stage 2 选出的段落停在"章节标题"级别（如"本次发行可转债的基本条款 1、..."），具体条款（9、转股价格修正；11、赎回条款）被 500 字符截断
- **修复**：Stage 1 不截断；Stage 2 prompt 截断提高到 2000 字符；添加 `expand_evidence` 自动补选后一段
- **效果**：条款正文进入 evidence

#### 5g. 多选题全部判"错误"导致答案为空

模型对所有选项判"错误"时 `answer=""`。但单选 (mcq/tf) 必须选一个。

**修复**：mcq fallback — 当 answer 为空时从 individual judgments 中重建（取置信度最高的选项）。
**效果**：多选题空答案仍存在（合理—全错时空=正确），单选择题全消灭。

**最终效果**：空答案 25→0。

---

### 问题 6：推理准确率低（~20%）

**难度：★★★★★（困难，未完全解决）**

**现象**：空答案全部消除后，提交评分仅 17 分（约 20 题正确 / 100 题）。

**诊断**：
- 证据质量已显著改善（经 7 轮修复）
- 模型在证据充分时仍给出错误判断
- `qwen-plus` 的金融推理能力不足——对复杂数值计算、跨文档比较、雷同选项辨析容易出错

**待优化方向**：
1. 换用 `qwen-max`（更强模型）
2. Prompt 加入领域-specific few-shot 示例
3. 结构化财务数据提取层（从文本中解析出 `{指标名: {年份: 数值}}`）
4. 跨题文档缓存（同一份文档被多道题引用时不重复检索）

---

### 问题 7：Windows 环境编码问题

**难度：★☆☆☆☆（简单但反复出现）**

Bash 环境下 GBK 编码无法输出 emoji（如 API 返回的 😊），导致多个调试脚本报 `UnicodeEncodeError`。

**解决**：所有脚本添加 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`。

---

### 问题 8：stage1_retrieve 截断导致候选段不完整

**难度：★★☆☆☆（简单）**

`truncate_paragraphs(paragraphs, max_tokens=500)` 对 Stage 1 返回的候选段截断 500 字符。对于"基本条款"这类长段落，关键条款内容在截断点之后完全丢失。

**解决**：取消 Stage 1 截断，仅在 Stage 2 构建 LLM prompt 时截断 2000 字符。

---

## 四、问题难度分布总览

| 问题 | 难度 | 状态 |
|---|---|---|
| 中文分词缺失导致关键词失效 | ★★★★★ | ✅ 已解决 |
| 财务表格标签/数值分离 | ★★★★★ | ✅ 已解决 |
| 第一文档独占检索配额 | ★★★★☆ | ✅ 已解决 |
| 多指标评分排序 | ★★★★☆ | ✅ 已解决 |
| 证据段落截断丢失关键内容 | ★★★☆☆ | ✅ 已解决 |
| 多选题全错导致答案为空 | ★★★☆☆ | ✅ 已解决 |
| 模型推理准确率低 | ★★★★★ | 🔲 待解决 |
| MinerU 性能问题 | ★★★☆☆ | 🔲 待解决 |
| API 模型名称混乱 | ★★★☆☆ | ✅ 已解决 |
| SDK 响应格式变更 | ★★☆☆☆ | ✅ 已解决 |
| 魔搭社区 API 超时 | ★★☆☆☆ | ✅ 已解决 |
| Windows GBK 编码 | ★☆☆☆☆ | ✅ 已解决 |

---

## 五、最终系统指标

| 指标 | 值 |
|---|---|
| 空答案率 | 0% (原始 25%) |
| Token 消耗 | 2,478,221 / 5,000,000 |
| TokenScore | 0.504 |
| 评分系数 | 0.851 |
| 运行时间 | ~47 分钟 / 100 题 |
| 提交得分 | **17 分**（~20% 准确率） |
| 测试覆盖 | 50 个单元测试，全部通过 |

---

## 六、代码文件清单

```
agent/
├── qwen_client.py       # Qwen API 客户端（百炼/魔搭双后端 + TokenCounter）
├── preprocessor.py      # PDF/HTML/TXT → .md (PyMuPDF + BeautifulSoup)
├── indexer.py           # 关键词倒排索引 + 章节树 + 文档摘要
├── domain_router.py     # 零 Token 领域路由 + prompt 选择
├── retriever.py         # Stage1 关键词 + Stage2 Qwen 精筛
├── reasoner.py          # 批量 CoT 推理 + JSON/Regex fallback
├── validator.py         # 答案标准化 + 置信度校验
└── pipeline.py          # 主流程编排
prompts/
├── stage2_filter.txt    # 精筛 prompt（指标锚定版）
├── reasoner_default.txt
├── reasoner_numerical.txt
├── reasoner_temporal.txt
└── reasoner_clause.txt
tests/                   # 50 个单元测试
```
