import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# llama-server — llama.cpp Windows nativo, Vulkan/RX580
# Se rodando no WSL, defina LLAMA_SERVER_HOST com o IP do gateway Windows
LLAMA_SERVER_HOST: str = os.getenv("LLAMA_SERVER_HOST", "localhost")
LLAMA_SERVER_PORT: int = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
LLAMA_MODEL: str = os.getenv("LLAMA_MODEL", "qwen3.5-9b")
LLAMA_GGUF: str = os.getenv("LLAMA_GGUF", r"F:\models\Qwen3.5-9B-Q4_K_M.gguf")

# Qdrant (Docker)
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION: str = os.getenv("COLLECTION", "maquina")

# Embeddings — nomic-embed-text-v1.5 via HuggingFace (768 dims, sem serviço externo)
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

# RAG
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "5"))

# Ingestão
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))

# TTS — arquivo de referência para clonagem de voz
REFERENCE_AUDIO: str = os.getenv("REFERENCE_AUDIO", "Gravando.m4a")
