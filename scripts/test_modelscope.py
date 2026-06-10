"""Test ModelScope API connectivity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QWEN_BACKEND"] = "modelscope"
os.environ["MODELSCOPE_API_KEY"] = "ms-165a851e-841f-4744-b949-892d0fa38251"
os.environ["QWEN_MODEL"] = "Qwen/Qwen3.5-35B-A3B"

from agent.qwen_client import QwenClient

client = QwenClient()
print(f"Backend: {client._backend}")
print(f"Model: {client.model}")

response = client.chat(
    messages=[{"role": "user", "content": "Say hello in one word"}],
    max_tokens=20,
)
print(f"Response: {response['content'][:100]}")
print(f"Tokens: in={response['input_tokens']} out={response['output_tokens']}")
print("OK")
