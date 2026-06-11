"""Qwen API client — supports 百炼 (DashScope) and 魔搭 (ModelScope) backends.
Set QWEN_BACKEND env var to "modelscope" or "dashscope" (default: dashscope).
"""
import os
from dataclasses import dataclass


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
    """Unified Qwen API client. Backend auto-detected from QWEN_BACKEND env var.

    DashScope (百炼): uses dashscope SDK, env DASHSCOPE_API_KEY
    ModelScope (魔搭): uses openai SDK, env MODELSCOPE_API_KEY
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        backend = os.environ.get("QWEN_BACKEND", "dashscope")
        self._backend = backend
        self.counter = TokenCounter()

        if backend == "modelscope":
            self.model = model or os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-35B-A3B")
            self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY", "")
            self._base_url = "https://api-inference.modelscope.cn/v1"
        else:
            self.model = model or os.environ.get("QWEN_MODEL", "qwen-max")
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
            self._base_url = None

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, timeout: int = 120) -> dict:
        """Send chat request and return parsed response with token counts."""
        if self._backend == "modelscope":
            return self._chat_modelscope(messages, temperature, max_tokens, timeout)
        else:
            return self._chat_dashscope(messages, temperature, max_tokens, timeout)

    def _chat_dashscope(self, messages, temperature, max_tokens, timeout):
        import dashscope
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
        from dashscope import Generation
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

        # Extract content
        output = response.output
        if isinstance(output, dict):
            content = output.get("text", "") or ""
        elif hasattr(output, "choices") and output.choices:
            content = output.choices[0].message.content
        elif hasattr(output, "text"):
            content = output.text or ""
        else:
            content = ""

        # Extract token counts
        usage = response.usage
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

        self.counter.add(input_tokens, output_tokens)
        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def _chat_modelscope(self, messages, temperature, max_tokens, timeout):
        from openai import OpenAI
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self._base_url,
                timeout=float(timeout),
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(timeout),
            )
        except Exception as e:
            raise RuntimeError(f"ModelScope API call failed: {e}") from e

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        self.counter.add(input_tokens, output_tokens)
        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def reset_counter(self):
        self.counter = TokenCounter()
