import pytest
from agent.qwen_client import QwenClient, TokenCounter, build_chat_message


def test_build_chat_message():
    msg = build_chat_message("user", "hello")
    assert msg == {"role": "user", "content": "hello"}


def test_client_initializes_with_config(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    from agent.qwen_client import QwenClient
    client = QwenClient()
    assert client.model == "qwen-plus"
    assert client.api_key == "test-key"


def test_chat_returns_response_structure():
    from unittest.mock import patch, MagicMock
    from agent.qwen_client import QwenClient
    client = QwenClient(api_key="fake-key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.output = MagicMock()
    mock_response.output.choices = [
        MagicMock(message=MagicMock(content='{"answer": "A"}'))
    ]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    with patch("dashscope.Generation.call", return_value=mock_response):
        result = client.chat(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.1
        )
    assert result["content"] == '{"answer": "A"}'
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50


def test_token_counter_accumulates():
    counter = TokenCounter()
    counter.add(100, 50)
    counter.add(200, 30)
    assert counter.prompt_tokens == 300
    assert counter.completion_tokens == 80
    assert counter.total_tokens == 380
