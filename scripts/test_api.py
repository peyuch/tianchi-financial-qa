"""Quick API connectivity test."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.qwen_client import QwenClient

client = QwenClient()
print(f"Model: {client.model}")
print(f"API key: {client.api_key[:8]}...")

response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=30
)
print(f"Response: {response['content'][:200]}")
print(f"Tokens: in={response['input_tokens']} out={response['output_tokens']}")
print("API test: OK")
