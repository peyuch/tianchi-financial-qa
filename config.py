import os

# Paths
RAW_DIR = os.path.join(os.path.dirname(__file__), "public_dataset_upload", "raw")
QUESTIONS_DIR = os.path.join(os.path.dirname(__file__), "public_dataset_upload", "questions", "group_a")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "indexes")
SUMMARIES_DIR = os.path.join(os.path.dirname(__file__), "data", "summaries")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# 赛题基准模型: Qwen3.6-plus (DashScope上模型ID为 qwen-plus)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Domain configuration
DOMAIN_CONFIG = {
    "financial_reports": {
        "context_token_cap": 5000,
        "per_doc_cap": 2500,
        "max_docs": 2,
        "default_prompt": "reasoner_numerical",
    },
    "regulatory": {
        "context_token_cap": 2000,
        "per_doc_cap": 1000,
        "max_docs": 2,
        "default_prompt": "reasoner_clause",
    },
    "insurance": {
        "context_token_cap": 4000,
        "per_doc_cap": 1000,
        "max_docs": 4,
        "default_prompt": "reasoner_clause",
    },
    "financial_contracts": {
        "context_token_cap": 4000,
        "per_doc_cap": 2000,
        "max_docs": 2,
        "default_prompt": "reasoner_clause",
    },
    "research": {
        "context_token_cap": 3000,
        "per_doc_cap": 1500,
        "max_docs": 2,
        "default_prompt": "reasoner_default",
    },
}

# Doc ID → file path resolver
DOC_EXTENSION_MAP = {
    "insurance": [".pdf"],
    "financial_reports": [".PDF", ".pdf"],
    "financial_contracts": [".pdf"],
    "research": [".pdf"],
    "regulatory": [".txt", ".html", ".pdf"],
}

# Output CSV format (must match competition submission spec)
OUTPUT_CSV_FIELDNAMES = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]

# Token budget
TOKEN_BUDGET = 5_000_000

# Confidence threshold for triggering fallback
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
