import pytest
from unittest.mock import patch, MagicMock
from agent.reasoner import (
    build_reasoning_prompt,
    parse_reasoning_response,
    parse_reasoning_fallback,
    reason,
)


def test_build_reasoning_prompt_numerical():
    evidence = [{"doc_id": "doc1", "text": "2025年研发投入542亿元，营收7771亿元"}]
    question = "研发投入占营业收入的比例是上升还是下降？"
    options = {"A": "上升", "B": "下降"}
    prompt = build_reasoning_prompt("reasoner_numerical", evidence, question, options)
    assert "研发投入" in prompt
    assert '"answer"' in prompt


def test_parse_reasoning_response_valid_json():
    content = '''{
  "results": [
    {"option": "A", "judgment": "正确", "evidence": {"doc_id": "d1", "section": "s1", "quote": "q1", "reasoning": "test"}, "confidence": 0.9},
    {"option": "B", "judgment": "错误", "evidence": {"doc_id": "d1", "section": "s2", "quote": "q2", "reasoning": "test"}, "confidence": 0.8}
  ],
  "answer": "A"
}'''
    result = parse_reasoning_response(content, "mcq")
    assert result["answer"] == "A"
    assert len(result["results"]) == 2
    assert result["results"][0]["confidence"] == 0.9


def test_parse_reasoning_response_json_in_text():
    content = '''一些推理文字...
{
  "results": [{"option": "A", "judgment": "正确", "evidence": {"doc_id": "d1", "section": "s1", "quote": "x", "reasoning": "r"}, "confidence": 0.7}],
  "answer": "A"
}
更多文字...'''
    result = parse_reasoning_response(content, "mcq")
    assert result["answer"] == "A"


def test_parse_reasoning_fallback_no_json():
    content = "综合分析，选项A是正确的，选项B是错误的，因此答案为A。"
    result = parse_reasoning_fallback(content, "mcq")
    assert result["answer"] == "A"


def test_parse_reasoning_fallback_multi():
    content = "A正确，C也正确，答案是AC"
    result = parse_reasoning_fallback(content, "multi")
    assert result["answer"] == "AC"


def test_reason_calls_client(monkeypatch):
    from unittest.mock import patch, MagicMock
    from agent.qwen_client import QwenClient
    client = QwenClient(api_key="fake")
    evidence = [{"doc_id": "d1", "text": "test"}]
    question = "test question?"
    options = {"A": "yes", "B": "no"}

    mock_response = {
        "content": '{"results": [{"option": "A", "judgment": "正确", "evidence": {"doc_id": "d1", "section": "s1", "quote": "x", "reasoning": "r"}, "confidence": 0.9}], "answer": "A"}',
        "input_tokens": 500,
        "output_tokens": 100,
    }

    with patch.object(client, "chat", return_value=mock_response):
        result = reason(client, evidence, question, options, "reasoner_default", "mcq")
        assert result["answer"] == "A"
        assert result["input_tokens"] == 500
