"""Test available Qwen model names — with proper error handling."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashscope import Generation

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
models_to_try = ["qwen-plus", "qwen-max", "qwen-turbo", "qwen3.6-plus"]

for model in models_to_try:
    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            api_key=api_key,
        )
        print(f"  {model}: status={response.status_code}")
        if response.status_code == 200:
            if response.output and response.output.choices:
                content = response.output.choices[0].message.content
                print(f"    content: {content[:80]}")
                print(f"    usage: {response.usage}")
            else:
                print(f"    output is None or empty: {response.output}")
        else:
            print(f"    message: {response.message}")
            print(f"    code: {response.code}")
    except Exception as e:
        print(f"  {model}: EXCEPTION {type(e).__name__}: {e}")
    print()
