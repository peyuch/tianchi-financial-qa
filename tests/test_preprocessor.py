import os
import pytest
from agent.preprocessor import (
    resolve_doc_path,
    extract_text_pdf,
    extract_text_html,
    extract_text_txt,
    preprocess_document,
    DOMAIN_DIRS,
)


def test_resolve_doc_path_insurance():
    path = resolve_doc_path("insurance", "1")
    assert path is not None
    assert path.endswith("1.pdf")


def test_resolve_doc_path_regulatory_txt():
    path = resolve_doc_path(
        "regulatory",
        "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）"
    )
    assert path is not None
    assert path.endswith(".txt")


def test_resolve_doc_path_missing_returns_none():
    path = resolve_doc_path("insurance", "nonexistent_doc_999")
    assert path is None


def test_extract_text_txt():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("第一章 总则\n\n第一条 为了规范…\n第二条 本办法所称…")
        tmp_path = f.name
    try:
        text = extract_text_txt(tmp_path)
        assert "第一章 总则" in text
        assert "第一条" in text
    finally:
        os.unlink(tmp_path)


def test_extract_text_html():
    html_content = """<html><body>
    <h1>上市公司治理准则</h1>
    <div class="article"><p>第一条 为了规范上市公司运作…</p></div>
    </body></html>"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html_content)
        tmp_path = f.name
    try:
        text = extract_text_html(tmp_path)
        assert "上市公司治理准则" in text
        assert "第一条" in text
    finally:
        os.unlink(tmp_path)


def test_preprocess_document_outputs_markdown():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("第一章 总则\n\n第一条 测试内容。")
        tmp_path = f.name
    try:
        result = preprocess_document(tmp_path, "txt")
        assert result is not None
        assert result.endswith(".md")
    finally:
        os.unlink(tmp_path)
