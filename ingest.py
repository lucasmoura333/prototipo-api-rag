import qdrant_client
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DATA_DIR, EMBED_MODEL, QDRANT_HOST, QDRANT_PORT
from readers import load_pdfs, load_sheets, load_videos


def ingest() -> None:
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    docs: list[Document] = []

    pdf_files = sorted(DATA_DIR.glob("**/*.pdf"))
    if pdf_files:
        print(f"[ingest] Loading {len(pdf_files)} PDF(s)...")
        docs.extend(load_pdfs(pdf_files))

    sheet_files = sorted(DATA_DIR.glob("**/*.csv")) + sorted(DATA_DIR.glob("**/*.xlsx"))
    if sheet_files:
        print(f"[ingest] Loading {len(sheet_files)} spreadsheet(s)...")
        docs.extend(load_sheets(sheet_files))

    video_exts = ["**/*.mp4", "**/*.mkv", "**/*.avi", "**/*.mov"]
    video_files = [f for ext in video_exts for f in sorted(DATA_DIR.glob(ext))]
    if video_files:
        print(f"[ingest] Loading {len(video_files)} video(s)...")
        docs.extend(load_videos(video_files))

    text_files = sorted(DATA_DIR.glob("**/*.txt")) + sorted(DATA_DIR.glob("**/*.md"))
    if text_files:
        print(f"[ingest] Loading {len(text_files)} text file(s)...")
        reader = SimpleDirectoryReader(input_files=[str(f) for f in text_files])
        docs.extend(reader.load_data())

    if not docs:
        print("[ingest] No documents found in ./data/ — add files and retry.")
        return

    print(f"[ingest] Indexing {len(docs)} document(s)...")
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )
    print(f"[ingest] Done. {len(docs)} documents indexed into collection '{COLLECTION}'.")


if __name__ == "__main__":
    ingest()
