"""Debug API response structure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import Generation

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "say hi"}],
    max_tokens=20,
    api_key=api_key,
    timeout=30,
)
print("status_code:", response.status_code)
print("message:", response.message)
print("code:", response.code)
print()
print("output type:", type(response.output))
print("output:", response.output)
if response.output:
    print("choices type:", type(response.output.choices))
    if response.output.choices:
        print("choices[0]:", response.output.choices[0])
        if response.output.choices[0].message:
            print("message:", response.output.choices[0].message)
            print("content:", repr(response.output.choices[0].message.content))
print()
print("usage type:", type(response.usage))
print("usage:", response.usage)
