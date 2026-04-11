import os
import shutil

import qdrant_client
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel

from config import (
    COLLECTION,
    EMBED_MODEL,
    LLAMA_MODEL,
    LLAMA_SERVER_HOST,
    LLAMA_SERVER_PORT,
    QDRANT_HOST,
    QDRANT_PORT,
    SIMILARITY_TOP_K,
)

app = FastAPI(title="RAGatanga API", version="0.2.0")

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".md", ".mp4", ".mkv", ".avi", ".mov"}


def _build_llm() -> OpenAILike:
    return OpenAILike(
        model=LLAMA_MODEL,
        api_base=f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
        api_key="not-needed",
        is_chat_model=True,
        request_timeout=120.0,
        context_window=4096,
    )


_client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_vector_store = QdrantVectorStore(client=_client, collection_name=COLLECTION)
_llm = _build_llm()
_embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)

_index = VectorStoreIndex.from_vector_store(_vector_store, embed_model=_embed_model)
_query_engine = _index.as_query_engine(
    llm=_llm,
    similarity_top_k=SIMILARITY_TOP_K,
    response_mode="simple_summarize",
)


class QueryRequest(BaseModel):
    q: str
    refine: bool = True


class QueryResponse(BaseModel):
    response: str
    sources: list[str] = []
    refined: bool = False


class StudyAudioRequest(BaseModel):
    topic: str
    language: str = "pt"
    max_words: int = 400


_NARRATION_PROMPT = """\
Você é um professor explicando um tema de forma didática para um aluno.
Com base no contexto extraído do documento abaixo, escreva uma narração em áudio \
sobre o tema "{topic}". A narração deve:
- Ser em {language_name}
- Ter no máximo {max_words} palavras
- Usar linguagem natural e conversacional (será lida em voz alta)
- Evitar listas, bullets, markdown ou símbolos especiais
- Ir direto ao ponto, sem introduções do tipo "Olá, hoje vamos falar sobre..."

CONTEXTO DO DOCUMENTO:
{context}

NARRAÇÃO:"""


@app.get("/health")
def health():
    return {"status": "ok", "llm_backend": "llama-server", "model": LLAMA_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    result = _query_engine.query(body.q)
    sources = list({node.metadata.get("file_name", "unknown") for node in result.source_nodes})
    response_text = str(result)

    if body.refine:
        from refiner import refinar

        context = "\n---\n".join(node.get_content() for node in result.source_nodes)
        response_text = refinar(response_text, context)

    return QueryResponse(response=response_text, sources=sources, refined=body.refine)


@app.post("/ingest")
def ingest_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    os.makedirs("./data", exist_ok=True)
    dest = f"./data/{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": f"File '{file.filename}' saved to ./data/. Run `python ingest.py` to index it."
    }


_LANGUAGE_NAMES = {
    "pt": "português",
    "en": "English",
    "es": "español",
    "fr": "français",
    "de": "Deutsch",
}


@app.post("/study-audio", summary="Gera áudio de estudo com voz clonada do Gravando.m4a")
async def study_audio(body: StudyAudioRequest):
    """
    Pipeline completo:
    1. Busca conteúdo relevante no Qdrant (RAG)
    2. Pede ao LLM para gerar uma narração educacional
    3. Sintetiza o áudio com XTTS v2 clonando a voz de REFERENCE_AUDIO
    4. Retorna o WAV gerado

    Parâmetros:
    - topic: tema / pergunta sobre o documento
    - language: código ISO (pt, en, es, fr, de) — padrão pt
    - max_words: tamanho máximo da narração em palavras — padrão 400
    """
    from tts_engine import generate_audio

    # 1. RAG: busca os chunks mais relevantes
    result = _query_engine.query(body.topic)
    context = "\n\n".join(node.get_content() for node in result.source_nodes)
    sources = list({node.metadata.get("file_name", "unknown") for node in result.source_nodes})

    if not context.strip():
        raise HTTPException(status_code=404, detail="Nenhum conteúdo relevante encontrado para o tema.")

    # 2. LLM: gera narração conversacional
    lang_name = _LANGUAGE_NAMES.get(body.language, body.language)
    prompt = _NARRATION_PROMPT.format(
        topic=body.topic,
        language_name=lang_name,
        max_words=body.max_words,
        context=context[:3000],  # evita overflow de contexto
    )
    narration = str(_llm.complete(prompt)).strip()

    if not narration:
        raise HTTPException(status_code=500, detail="LLM não gerou narração.")

    # 3. TTS: sintetiza com voz clonada
    try:
        audio_path = generate_audio(narration, language=body.language)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Retorna o WAV — FileResponse serve o arquivo e o deleta é responsabilidade do cliente
    filename = f"estudo_{body.topic[:30].replace(' ', '_')}.wav"
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename,
        headers={"X-Sources": ", ".join(sources), "X-Narration-Words": str(len(narration.split()))},
    )
