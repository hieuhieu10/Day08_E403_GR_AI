"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import sys
from pathlib import Path

# Console Windows mặc định dùng cp1252 → không in được ký tự ✓/tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"
COLLECTION_NAME = "DrugLawDocs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking strategy:
# - Chọn RecursiveCharacterTextSplitter vì nó an toàn & phổ biến nhất, hoạt động
#   tốt với CẢ văn bản luật (đoạn dài) lẫn bài báo (đã có rác/menu lẫn lộn).
#   Nó cắt theo thứ tự ưu tiên \n\n → \n → câu → từ, nên giữ được ngữ nghĩa
#   tốt hơn cắt cứng theo số ký tự.
CHUNK_SIZE = 500        # ~500 ký tự ≈ 1 đoạn ngắn: đủ ngữ cảnh cho 1 ý nhưng
                        # vẫn nhỏ để embedding chính xác & không loãng khi search.
CHUNK_OVERLAP = 50      # Lặp 50 ký tự (10%) giữa 2 chunk để không cắt đứt câu/ý
                        # nằm ở ranh giới chunk → tránh mất ngữ cảnh.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model:
# - BAAI/bge-m3: multilingual, hỗ trợ tiếng Việt rất tốt, 1024 chiều, là một
#   trong những model open-source mạnh nhất cho retrieval đa ngôn ngữ. Phù hợp
#   vì dữ liệu (luật + báo) hoàn toàn bằng tiếng Việt.
# - Lưu ý: lần chạy đầu sẽ tải model (~2GB) về máy.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Vector store:
# - ChromaDB: local, không cần Docker, lưu thẳng ra đĩa (persistent). Đơn giản,
#   nhẹ, đủ cho quy mô bài tập. Dense search; phần lexical (BM25) làm ở Task 6
#   rồi kết hợp thành hybrid ở Task 9.
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Đang tải model {EMBEDDING_MODEL} (lần đầu có thể ~2GB)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # chuẩn hoá để dùng cosine similarity nhất quán
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Tạo lại collection từ đầu để tránh trùng lặp khi chạy nhiều lần.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # chưa tồn tại thì bỏ qua

    # embedding_function=None vì ta tự cung cấp vector đã tính sẵn (bge-m3).
    # Đặt cosine làm hàm khoảng cách cho khớp với embedding đã normalize.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Chroma giới hạn batch size → chia nhỏ cho an toàn.
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=documents[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"  Collection '{COLLECTION_NAME}' hiện có {collection.count()} chunks.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
