"""Qwen API client via 百炼 DashScope."""
import os
from dataclasses import dataclass
from dashscope import Generation


def _extract_content(response) -> str:
    """Extract content from response, handling both old and new SDK formats."""
    output = response.output
    # New format: output is a dict with "text" key
    if isinstance(output, dict):
        return output.get("text", "") or ""
    # Old format: output.choices[0].message.content
    if hasattr(output, "choices") and output.choices:
        return output.choices[0].message.content
    # Try .text attribute
    if hasattr(output, "text"):
        return output.text or ""
    return ""


def _extract_tokens(response, kind: str) -> int:
    """Extract token count from response usage."""
    usage = response.usage
    # New format: usage is a dict
    if isinstance(usage, dict):
        return usage.get(f"{kind}_tokens", 0)
    # Old format: usage.input_tokens / usage.output_tokens
    return getattr(usage, f"{kind}_tokens", 0)


@dataclass
class TokenCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def build_chat_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class QwenClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("QWEN_MODEL", "qwen-plus")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.counter = TokenCounter()

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, timeout: int = 120) -> dict:
        """Send chat request and return parsed response with token counts."""
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Qwen API call failed: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen API error {response.status_code}: {response.message}"
            )

        content = _extract_content(response)
        input_tokens = _extract_tokens(response, "input")
        output_tokens = _extract_tokens(response, "output")

        self.counter.add(input_tokens, output_tokens)

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def reset_counter(self):
        self.counter = TokenCounter()
