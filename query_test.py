import time

import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.qdrant import QdrantVectorStore

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

QUERIES = [
    "Summarize the main topics covered in the documents.",
    "What are the key technical specifications mentioned?",
    "List the most important data points found in the files.",
]


def main() -> None:
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llm = OpenAILike(
        model=LLAMA_MODEL,
        api_base=f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1",
        api_key="not-needed",
        is_chat_model=True,
        request_timeout=120.0,
        context_window=4096,
    )
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=SIMILARITY_TOP_K,
                                         response_mode="simple_summarize",
                                         streaming=True)

    for q in QUERIES:
        print(f"\n{'='*60}")
        print(f"[QUERY] {q}")
        print("[RESPONSE] ", end="", flush=True)
        t0 = time.time()
        token_count = 0
        streaming_response = query_engine.query(q)
        for token in streaming_response.response_gen:
            print(token, end="", flush=True)
            token_count += 1
        elapsed = time.time() - t0
        sources = list({n.metadata.get("file_name", "?") for n in streaming_response.source_nodes})
        print(f"\n[{token_count} tokens | {elapsed:.1f}s | {token_count/elapsed:.1f} tok/s]")
        print(f"[SOURCES] {sources}")


if __name__ == "__main__":
    main()
