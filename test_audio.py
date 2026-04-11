"""Test /study-audio endpoint - espera JSON com narration + audio_url."""
import urllib.request, json, time

payload = json.dumps({
    "topic": "especificacoes tecnicas do dispositivo",
    "language": "pt",
    "max_words": 100
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/study-audio",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

print("Chamando /study-audio (1a vez baixa modelo TTS ~900MB)...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    elapsed = round(time.time()-t0, 1)
    print(f"OK em {elapsed}s")
    print(f"audio_url: {data.get('audio_url')}")
    print(f"word_count: {data.get('word_count')}")
    print(f"sources: {data.get('sources')}")
    print(f"\nNarração:\n{data.get('narration','')[:400]}")
except Exception as e:
    elapsed = round(time.time()-t0, 1)
    print(f"FALHOU em {elapsed}s: {e}")

