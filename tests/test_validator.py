import pytest
from agent.validator import (
    normalize_answer,
    validate_confidence,
    format_output_row,
    build_evidence_entry,
)


def test_normalize_mcq_single_letter():
    assert normalize_answer("A", "mcq") == "A"
    assert normalize_answer("  b  ", "mcq") == "B"
    assert normalize_answer("答案是C", "mcq") == "C"


def test_normalize_multi_sort_dedup():
    assert normalize_answer("CAB", "multi") == "ABC"
    assert normalize_answer("A,A,B,C", "multi") == "ABC"
    assert normalize_answer("c a b", "multi") == "ABC"


def test_normalize_tf():
    assert normalize_answer("A", "tf") == "A"
    assert normalize_answer("正确", "tf") == "A"


def test_normalize_empty():
    assert normalize_answer("", "mcq") == ""
    assert normalize_answer("xyz_test_789", "mcq") == ""


def test_normalize_multi_filters_out_of_range():
    assert normalize_answer("ABCE", "multi") == "ABC"
    assert normalize_answer("ABCDE", "multi") == "ABCD"
    assert normalize_answer("XYZ", "multi") == ""


def test_validate_confidence_high():
    results = [
        {"option": "A", "confidence": 0.9},
        {"option": "B", "confidence": 0.85},
    ]
    assert validate_confidence(results) is True


def test_validate_confidence_low():
    results = [
        {"option": "A", "confidence": 0.4},
        {"option": "B", "confidence": 0.9},
    ]
    assert validate_confidence(results) is False


def test_validate_confidence_all_high():
    results = [
        {"option": "A", "confidence": 0.95},
        {"option": "B", "confidence": 0.88},
        {"option": "C", "confidence": 0.92},
        {"option": "D", "confidence": 0.81},
    ]
    assert validate_confidence(results) is True
