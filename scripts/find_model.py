"""Test available Qwen model names."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import Generation

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
models_to_try = ["qwen-plus", "qwen-max", "qwen-turbo", "qwen3.6-plus", "qwen2.5-72b-instruct"]

for model in models_to_try:
    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            api_key=api_key,
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            print(f"  {model}: OK -> {content[:50]}")
        else:
            print(f"  {model}: FAIL ({response.status_code}: {response.message})")
    except Exception as e:
        print(f"  {model}: ERROR ({e})")
