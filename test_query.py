import urllib.request
import json
import time


def query(q, refine=False):
    body = json.dumps({"q": q, "refine": refine}).encode()
    req = urllib.request.Request(
        "http://localhost:8000/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        elapsed = round(time.time() - t, 1)
        print(f"[refine={refine}] {elapsed}s")
        print(data["response"])
        print()
    except Exception as e:
        elapsed = round(time.time() - t, 1)
        print(f"[refine={refine}] ERRO {elapsed}s: {e}")


query("Qual o tema dos documentos?", refine=False)
query("Qual o tema dos documentos?", refine=True)
