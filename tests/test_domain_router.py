import pytest
from agent.domain_router import (
    route_domain,
    select_reasoning_prompt,
    PROMPT_SELECTION_RULES,
    get_output_format,
)


def test_route_domain_financial_reports():
    config = route_domain("financial_reports")
    assert config["context_token_cap"] == 5000
    assert config["per_doc_cap"] == 2500
    assert config["default_prompt"] == "reasoner_numerical"


def test_route_domain_insurance():
    config = route_domain("insurance")
    assert config["per_doc_cap"] == 1000
    assert config["max_docs"] == 4


def test_route_domain_unknown_returns_default():
    config = route_domain("unknown_domain")
    assert config is not None
    assert "context_token_cap" in config


def test_select_temporal_prompt():
    prompt = select_reasoning_prompt(
        "financial_reports",
        "2025年营业收入较2024年实现增长，净利润同比出现下滑"
    )
    assert prompt in ("reasoner_temporal", "reasoner_numerical")


def test_select_numerical_prompt():
    prompt = select_reasoning_prompt(
        "financial_reports",
        "计算各产品身故保险金金额并排序"
    )
    assert prompt == "reasoner_numerical"


def test_select_default_prompt():
    prompt = select_reasoning_prompt(
        "regulatory",
        "关于金融机构客户受益所有人识别管理办法的实施要求"
    )
    assert prompt == "reasoner_clause"


def test_select_from_question_json():
    question = {
        "domain": "financial_reports",
        "question": "根据比亚迪连续两年的年度报告，下列关于公司经营业绩变化的描述中，哪些是准确的？",
        "answer_format": "multi",
        "doc_ids": ["annual_byd_2024_report", "annual_byd_2025_report"],
    }
    domain_config = route_domain(question["domain"])
    prompt_name = select_reasoning_prompt(question["domain"], question["question"])
    assert domain_config["context_token_cap"] == 5000
    assert prompt_name in ("reasoner_temporal", "reasoner_numerical")
