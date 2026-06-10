"""Debug API response structure — explore all fields."""
import os, sys
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
out = response.output
if out:
    print("output fields:", [f for f in dir(out) if not f.startswith("_")])
    print("text attr:", getattr(out, "text", "N/A"))
    if hasattr(out, "choices") and out.choices:
        print("choices:", out.choices)
    # Try other common patterns
    for attr in ["text", "content", "message", "result", "data"]:
        val = getattr(out, attr, None)
        if val is not None:
            print(f"  .{attr}: {repr(str(val)[:100])}")
