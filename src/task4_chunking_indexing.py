from pathlib import Path
import chromadb
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKING_METHOD = "markdown_header"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"


def load_documents() -> list[dict]:
    documents = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        doc_type = "legal" if "legal" in str(md_file).lower() else "news"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type
            }
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    global_chunk_index = 0 

    for doc in documents:
        header_docs = md_splitter.split_text(doc["content"])

        for header_doc in header_docs:
            sub_chunks = recursive_splitter.split_text(header_doc.page_content)

            for text in sub_chunks:
                chunks.append({
                    "content": text,
                    "metadata": {
                        **doc["metadata"],
                        **header_doc.metadata,
                        "chunk_index": global_chunk_index,
                    },
                })
                global_chunk_index += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    assert len(chunks[0]["embedding"]) == EMBEDDING_DIM

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    client = chromadb.PersistentClient(path="./chromadb")

    collection_name = "DrugLawDocs"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        collection.add(
            ids=[
                f"{c['metadata']['source']}_{c['metadata']['chunk_index']}"
                for c in batch
            ],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[
                {
                    "source": c["metadata"]["source"],
                    "doc_type": c["metadata"]["type"],
                    "chunk_index": c["metadata"]["chunk_index"],
                }
                for c in batch
            ]
        )

    print(f"Saved {collection.count()} chunks to ChromaDB")


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"Chunking: {CHUNKING_METHOD}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Done")


if __name__ == "__main__":
    run_pipeline()