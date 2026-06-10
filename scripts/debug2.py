"""Debug API response structure — handle encoding."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import Generation

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "say hi"}],
    max_tokens=20,
    api_key=api_key,
)
print("status:", response.status_code)
if response.output:
    print("choices count:", len(response.output.choices) if response.output.choices else 0)
    if response.output.choices:
        msg = response.output.choices[0].message
        content = msg.content
        print("content (repr):", repr(content[:50]))
        print("content (ascii):", content.encode("ascii", errors="replace").decode("ascii")[:50])
    print("usage:", response.usage)
    print("input_tokens:", response.usage.input_tokens)
    print("output_tokens:", response.usage.output_tokens)
else:
    print("output is None")
