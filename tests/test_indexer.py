import json
import os
import pytest
from agent.indexer import (
    build_keyword_index,
    extract_headings,
    extract_summary,
    index_document,
    search_keyword_index,
    CLAUSE_PATTERN,
    ENTITY_PATTERNS,
)

@pytest.fixture
def sample_md():
    return """# 上市公司治理准则

## 第一章 总则

第一条 为了规范上市公司运作，保护投资者合法权益，根据《中华人民共和国公司法》、《中华人民共和国证券法》和其他有关规定，制定本准则。

第二条 本准则适用于依照中国法律在中国境内设立并在证券交易所上市的股份有限公司。

## 第二章 股东与股东大会

### 第一节 股东权利

第十条 上市公司股东享有法律、行政法规和公司章程规定的合法权利。

## 第三章 董事与董事会

第二十条 董事会成员应当具备履行职责所必需的知识、技能和素质。
"""

def test_extract_headings(sample_md):
    headings = extract_headings(sample_md)
    assert len(headings) >= 3
    assert headings[0]["level"] == 1
    assert headings[0]["text"] == "上市公司治理准则"
    assert headings[1]["text"] == "第一章 总则"

def test_clause_pattern_finds_articles():
    matches = CLAUSE_PATTERN.findall("第一条 为了规范上市公司运作…")
    assert len(matches) > 0

def test_build_keyword_index(sample_md):
    idx = build_keyword_index("test_doc", sample_md)
    assert "test_doc" in idx
    assert any("第一条" in entry["text"] for entry in idx["test_doc"])

def test_search_keyword_index():
    idx = {
        "doc1": [
            {"para_id": 0, "text": "第四十七条 公司下列对外担保行为，须经股东会审议通过。"},
            {"para_id": 1, "text": "其他无关内容。"},
        ],
        "doc2": [
            {"para_id": 0, "text": "第八十二条 下列事项由股东会以特别决议通过。"},
        ],
    }
    results = search_keyword_index(idx, "第四十七条")
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"

    results = search_keyword_index(idx, "股东会")
    assert len(results) >= 2

    results = search_keyword_index(idx, "nonexistent_term")
    assert len(results) == 0

def test_extract_summary(sample_md):
    summary = extract_summary("test_doc", "regulatory", sample_md)
    assert summary["doc_id"] == "test_doc"
    assert summary["domain"] == "regulatory"
    assert len(summary["summary_keywords"]) > 0
    assert len(summary["toc"]) > 0
    assert "上市公司治理准则" in summary["toc"][0]
