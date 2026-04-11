"""Quick test: call llama-server directly from WSL with 5-minute timeout."""
import time, urllib.request, json

payload = json.dumps({
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "/no_think Qual e o tema principal do documento? Resposta curta."}],
    "max_tokens": 80
}).encode()

url = "http://localhost:8080/v1/chat/completions"
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

print(f"Sending request to {url} ...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.load(resp)
    elapsed = time.time() - t0
    content = body["choices"][0]["message"]["content"]
    thinking = body["choices"][0]["message"].get("reasoning_content", "")
    finish = body["choices"][0].get("finish_reason", "?")
    print(f"OK in {elapsed:.1f}s | finish={finish}")
    print(f"Content: {repr(content[:200])}")
    print(f"Thinking (first 200c): {repr(thinking[:200])}")
    print(f"Full choice keys: {list(body['choices'][0]['message'].keys())}")
    print(f"Usage: {body.get('usage', {})}")
except Exception as e:
    print(f"FAILED after {time.time()-t0:.1f}s: {e}")
