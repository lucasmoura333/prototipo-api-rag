#!/bin/bash
set -e

PROJECT=/mnt/f/Eu/LucasMoura333/Projetos/prototipo-api-rag
cd "$PROJECT"
source .venv/bin/activate

echo "=== Testando velocidade do modelo (prompt mínimo) ==="
curl -s -X POST http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Responda apenas: OK"}],"stream":false,"think":false}' \
  -o /tmp/resp.json

python3 - <<'PY'
import json
d = json.load(open('/tmp/resp.json'))
content = d.get('message', {}).get('content', 'ERRO')
t = d.get('total_duration', 0) // 1_000_000_000
eval_count = d.get('eval_count', 0)
eval_ns = d.get('eval_duration', 1)
tps = eval_count / (eval_ns / 1e9) if eval_ns else 0
print(f"Resposta : {content[:120]}")
print(f"Tempo    : {t}s")
print(f"Tokens/s : {tps:.1f}")
PY

echo ""
echo "=== Ingerindo PDF 2.pdf ==="
python3 ingest.py

echo ""
echo "=== Query direta sobre o PDF ==="
python3 - <<'PY'
import os
import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="maquina")
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
llm = Ollama(
    model="qwen3.5:9b",
    request_timeout=600.0,
    context_window=4096,
    system_prompt="/no_think",
)

import time

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
# streaming=True — tokens aparecem conforme são gerados
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    response_mode="simple_summarize",
    streaming=True,
)

perguntas = [
    "Qual é o assunto principal deste documento?",
    "Quais são os pontos mais importantes mencionados?",
]

for p in perguntas:
    print(f"\n{'='*50}")
    print(f"PERGUNTA: {p}")
    print("RESPOSTA: ", end="", flush=True)
    t0 = time.time()
    token_count = 0
    streaming_response = query_engine.query(p)
    for token in streaming_response.response_gen:
        print(token, end="", flush=True)
        token_count += 1
    elapsed = time.time() - t0
    fontes = list({n.metadata.get("file_name", "?") for n in streaming_response.source_nodes})
    print(f"\n[{token_count} tokens em {elapsed:.1f}s = {token_count/elapsed:.1f} tok/s]")
    print(f"FONTES  : {fontes}")
PY
