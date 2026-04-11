import json
import sys
import time

import requests

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "id": "T01",
        "query": "What are the main topics covered in the documents?",
        "must_contain": [],
        "must_not_contain": ["i don't know", "i cannot find", "no information"],
    },
    {
        "id": "T02",
        "query": "List the key technical specifications mentioned.",
        "must_contain": [],
        "must_not_contain": ["i don't know"],
    },
    {
        "id": "T03",
        "query": "Summarize the most important data points from the spreadsheets.",
        "must_contain": [],
        "must_not_contain": [],
    },
]


def run_tests(refine: bool = True) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Running {len(TEST_CASES)} test(s) — refine={refine}")
    print(f"{'='*60}\n")

    results = []
    for tc in TEST_CASES:
        start = time.time()
        try:
            resp = requests.post(
                f"{BASE_URL}/query",
                json={"q": tc["query"], "refine": refine},
                timeout=300,
            )
            resp.raise_for_status()
            elapsed = time.time() - start
            data = resp.json()
            response_text = data.get("response", "")

            contains_ok = all(
                kw.lower() in response_text.lower() for kw in tc["must_contain"]
            )
            contains_bad = any(
                kw.lower() in response_text.lower() for kw in tc["must_not_contain"]
            )
            passed = contains_ok and not contains_bad

            results.append(
                {
                    "id": tc["id"],
                    "passed": passed,
                    "latency_s": round(elapsed, 2),
                    "sources": data.get("sources", []),
                    "refined": data.get("refined", False),
                    "response_preview": response_text[:300],
                }
            )
            status = "PASS" if passed else "FAIL"
            print(f"[{tc['id']}] {status} — {elapsed:.1f}s | sources: {data.get('sources', [])}")
            print(f"  {response_text[:200]}\n")

        except Exception as e:
            elapsed = time.time() - start
            results.append(
                {"id": tc["id"], "passed": False, "error": str(e), "latency_s": round(elapsed, 2)}
            )
            print(f"[{tc['id']}] ERROR — {e}\n")

    passed_count = sum(1 for r in results if r.get("passed"))
    print(f"\n{'='*60}")
    print(f"Result: {passed_count}/{len(results)} passed")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    use_refine = "--no-refine" not in sys.argv
    run_tests(refine=use_refine)
