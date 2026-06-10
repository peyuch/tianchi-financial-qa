import json
import pytest
from unittest.mock import patch, MagicMock
from agent.pipeline import (
    load_questions,
    collect_doc_list,
    run_pipeline,
)


@pytest.fixture
def sample_question():
    return {
        "qid": "test_001",
        "domain": "regulatory",
        "split": "A",
        "question": "关于测试法规的实施要求，下列说法正确的有？",
        "options": {"A": "选项A内容", "B": "选项B内容"},
        "answer_format": "multi",
        "type": "多选题",
        "doc_ids": ["strict_test_001"],
    }


def test_load_questions(tmp_path):
    q_file = tmp_path / "test_questions.json"
    q_file.write_text(json.dumps([
        {"qid": "q1", "domain": "regulatory", "question": "test?",
         "options": {"A": "a", "B": "b"}, "answer_format": "multi",
         "type": "test", "doc_ids": ["d1"]}
    ], ensure_ascii=False))
    questions = load_questions(str(q_file))
    assert len(questions) == 1
    assert questions[0]["qid"] == "q1"


def test_collect_doc_list(sample_question):
    questions = [sample_question]
    doc_list = collect_doc_list(questions)
    assert ("regulatory", "strict_test_001") in doc_list


def test_run_pipeline_function_exists():
    from agent.pipeline import run_pipeline
    assert callable(run_pipeline)
