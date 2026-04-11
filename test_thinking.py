"""Test llama-server props and thinking mode disable options."""
import urllib.request, json

# Check /props
req = urllib.request.Request("http://localhost:8080/props")
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        props = json.load(r)
    print("=== /props ===")
    for k, v in props.items():
        if k != "default_generation_settings":
            print(f"  {k}: {v}")
    gs = props.get("default_generation_settings", {})
    print(f"  n_predict: {gs.get('n_predict')}")
    print(f"  total_slots: {props.get('total_slots')}")
except Exception as e:
    print(f"props error: {e}")

# Test thinking=false via chat_template_kwargs
print("\n=== Test chat_template_kwargs enable_thinking=False ===")
import time
payload = json.dumps({
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Diga apenas: ola mundo"}],
    "max_tokens": 50,
    "chat_template_kwargs": {"enable_thinking": False}
}).encode()

req2 = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)
t0 = time.time()
try:
    with urllib.request.urlopen(req2, timeout=60) as r:
        body = json.load(r)
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    print(f"  OK in {elapsed:.1f}s | finish={body['choices'][0].get('finish_reason')}")
    print(f"  content: {repr(msg.get('content', ''))}")
    print(f"  reasoning_content: {repr(msg.get('reasoning_content','')[:100])}")
    print(f"  usage: {body.get('usage', {})}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test no_think via prompt
print("\n=== Test /no_think prefix ===")
payload2 = json.dumps({
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "/no_think Diga apenas: ola mundo"}],
    "max_tokens": 50
}).encode()
req3 = urllib.request.Request(
    "http://localhost:8080/v1/chat/completions",
    data=payload2,
    headers={"Content-Type": "application/json"}
)
t0 = time.time()
try:
    with urllib.request.urlopen(req3, timeout=60) as r:
        body = json.load(r)
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    print(f"  OK in {elapsed:.1f}s | finish={body['choices'][0].get('finish_reason')}")
    print(f"  content: {repr(msg.get('content', ''))}")
    print(f"  reasoning_content: {repr(msg.get('reasoning_content','')[:100])}")
    print(f"  usage: {body.get('usage', {})}")
except Exception as e:
    print(f"  FAILED: {e}")
