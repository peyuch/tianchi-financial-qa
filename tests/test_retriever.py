import pytest
from agent.retriever import (
    stage1_retrieve,
    extract_keywords_from_question,
    truncate_paragraphs,
    allocate_per_doc,
    should_skip_stage2,
)

@pytest.fixture
def sample_index():
    return {
        "doc1": [
            {"para_id": 0, "text": "第四十七条 公司下列对外担保行为，须经股东会审议通过。"},
            {"para_id": 1, "text": "2025年营业收入为7771亿元，较2024年增长29%"},
            {"para_id": 2, "text": "独立董事不得与其所受聘上市公司存在利害关系。"},
        ],
        "doc2": [
            {"para_id": 0, "text": "第八十二条 下列事项由股东会以特别决议通过。"},
            {"para_id": 1, "text": "2025年研发投入为542亿元，占营业收入比例6.97%"},
        ],
    }

def test_stage1_retrieve_by_clause(sample_index):
    results = stage1_retrieve(sample_index, ["doc1", "doc2"], "第四十七条")
    assert len(results) > 0
    assert any("第四十七条" in r["text"] for r in results)

def test_stage1_retrieve_by_entity(sample_index):
    results = stage1_retrieve(sample_index, ["doc1", "doc2"], "营业收入")
    assert len(results) >= 2

def test_stage1_retrieve_no_match(sample_index):
    results = stage1_retrieve(sample_index, ["doc1", "doc2"], "nonexistent_term")
    assert len(results) == 0

def test_extract_keywords():
    question = "根据比亚迪连续两年的年度报告，公司营业收入较2024年实现增长"
    keywords = extract_keywords_from_question(question)
    assert "营业收入" in keywords
    assert "比亚迪" in keywords

def test_truncate_paragraphs(sample_index):
    paragraphs = [p for entries in sample_index.values() for p in entries]
    truncated = truncate_paragraphs(paragraphs, max_tokens=50)
    assert len(truncated) <= len(paragraphs)
    for p in truncated:
        assert len(p["text"]) <= 100

def test_allocate_per_doc():
    evidence = [
        {"doc_id": "doc1", "para_id": 0, "text": "A" * 60},
        {"doc_id": "doc1", "para_id": 1, "text": "B" * 60},
        {"doc_id": "doc2", "para_id": 0, "text": "C" * 60},
    ]
    result = allocate_per_doc(evidence, per_doc_cap=100, max_docs=4)
    assert len(result) >= 2  # at least one from each of 2 docs

def test_should_skip_stage2_direct_match():
    candidates = [
        {"doc_id": "d1", "para_id": 0, "text": "第四十七条 公司下列对外担保行为"},
    ]
    assert should_skip_stage2(candidates, "第四十七条") is True

def test_should_skip_stage2_no_match():
    candidates = [
        {"doc_id": "d1", "para_id": 0, "text": "一些无关内容"},
        {"doc_id": "d1", "para_id": 1, "text": "更多无关内容"},
        {"doc_id": "d2", "para_id": 0, "text": "还是无关"},
        {"doc_id": "d2", "para_id": 1, "text": "依然无关"},
        {"doc_id": "d2", "para_id": 2, "text": "最后无关"},
        {"doc_id": "d2", "para_id": 3, "text": "再加一个"},
    ]
    assert should_skip_stage2(candidates, "一些无关问题") is False
