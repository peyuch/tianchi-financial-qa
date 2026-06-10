"""Check dict keys and values."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import Generation

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Say one word"}],
    max_tokens=20,
    api_key=api_key,
)

out = response.output
print("keys:", list(out.keys()))
for k, v in out.items():
    val_str = str(v)[:100]
    print(f"  {k}: {val_str}")

print()
try:
    text = out["text"]
    print("output.text:", repr(text))
except:
    print("no 'text' key")
