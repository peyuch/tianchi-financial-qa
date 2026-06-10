"""Quick API test with fixed model name."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.qwen_client import QwenClient

client = QwenClient()
print(f"Model: {client.model}")

response = client.chat(
    messages=[{"role": "user", "content": "Say hello in one word"}],
    max_tokens=20
)
content = response["content"].encode("utf-8", errors="replace").decode("utf-8")
print(f"Response: {content}")
print(f"Tokens: in={response['input_tokens']} out={response['output_tokens']}")
print(f"Counter: prompt={client.counter.prompt_tokens} completion={client.counter.completion_tokens} total={client.counter.total_tokens}")
print("OK")
