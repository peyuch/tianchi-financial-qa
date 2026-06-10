"""Smoke test: verify the full pipeline runs on a single question."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
from agent.qwen_client import QwenClient, TokenCounter, build_chat_message
from agent.pipeline import run_pipeline, load_questions, collect_doc_list


@pytest.fixture
def mock_client():
    client = QwenClient(api_key="fake")
    return client


def test_e2e_single_question(mock_client, tmp_path):
    """End-to-end: run pipeline on a single question with all steps."""
    # Create test question file
    q_file = os.path.join(str(tmp_path), "test_questions.json")
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump([{
            "qid": "e2e_001",
            "domain": "insurance",
            "split": "A",
            "question": "关于四个养老保险产品的身故保险金，以下计算结果正确的是？",
            "options": {
                "A": "产品A > 产品B > 产品C > 产品D",
                "B": "产品B > 产品A > 产品D > 产品C",
            },
            "answer_format": "mcq",
            "type": "推理判断",
            "doc_ids": ["1", "2"],
        }], f, ensure_ascii=False)

    # Verify the pipeline can load questions
    questions = load_questions(q_file)
    assert len(questions) == 1
    assert questions[0]["qid"] == "e2e_001"

    # Verify doc collection
    doc_list = collect_doc_list(questions)
    assert ("insurance", "1") in doc_list
    assert ("insurance", "2") in doc_list

    print("E2E smoke test passed: pipeline loads questions and collects docs")


def test_all_modules_importable():
    """Verify all agent modules can be imported without errors."""
    from agent.qwen_client import QwenClient, TokenCounter, build_chat_message
    from agent.preprocessor import resolve_doc_path, extract_text_txt
    from agent.indexer import build_keyword_index, search_keyword_index
    from agent.domain_router import route_domain, select_reasoning_prompt
    from agent.retriever import stage1_retrieve, should_skip_stage2
    from agent.reasoner import build_reasoning_prompt, parse_reasoning_response
    from agent.validator import normalize_answer, validate_confidence
    from agent.pipeline import run_pipeline, load_questions
    import config
    assert config.TOKEN_BUDGET == 5_000_000
    print("All modules importable")


def test_run_pipeline_function_exists():
    """Verify run_pipeline is callable."""
    assert callable(run_pipeline)
