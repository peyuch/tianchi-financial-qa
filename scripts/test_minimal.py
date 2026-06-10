"""Minimal API test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("DASHSCOPE_API_KEY:", os.environ.get("DASHSCOPE_API_KEY", "NOT SET")[:10] + "...")

from dashscope import Generation
response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "say hi"}],
    max_tokens=10,
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
)
print("status_code:", response.status_code)
print("output:", response.output)
if response.status_code == 200 and response.output:
    print("text:", response.output.choices[0].message.content)
elif response.status_code != 200:
    print("message:", response.message)
