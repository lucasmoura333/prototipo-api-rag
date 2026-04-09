# prototipo-api-rag

Protótipo local de API com RAG — consulta de documentos (PDFs, planilhas, vídeos) via LLM, com refinamento final por Gemma 3.

## Stack

- **LLM base**: Qwen3 9B Q4_K_M (Ollama)
- **LLM refinamento**: Gemma 3 4B (Ollama)
- **RAG**: LlamaIndex
- **Vector DB**: Qdrant (Docker)
- **API**: FastAPI
- **GPU (teste)**: RX580 via Vulkan / ROCm

## Quick Start

```bash
# 1. Subir Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 2. Baixar modelos
ollama pull qwen3:9b-q4_K_M
ollama pull gemma3:4b
ollama pull nomic-embed-text

# 3. Instalar dependências
pip install fastapi uvicorn llama-index llama-index-vector-stores-qdrant llama-index-llms-ollama llama-index-embeddings-ollama

# 4. Ingerir documentos e subir API
python ingest.py
uvicorn main:app --reload
```

> Plano detalhado de execução em `plano/` (local, não versionado).
