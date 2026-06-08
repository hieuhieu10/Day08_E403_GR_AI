"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import sys

from task4_chunking_indexing import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)

# Console Windows mặc định dùng cp1252 → không in được ký tự tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

# Cache model & collection ở cấp module → load 1 lần, tái dùng cho mọi truy vấn
# (tránh tải lại bge-m3 ~2GB mỗi lần gọi semantic_search).
_model = None
_collection = None


def _get_model():
    """Lazy-load embedding model (cùng model với Task 4)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Lazy-connect tới Chroma collection đã index ở Task 4."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 1: Embed query bằng cùng model & cách normalize như Task 4.
    model = _get_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # Bước 2: Query Chroma bằng cosine distance.
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Chuyển distance → similarity và trả về (đã sort sẵn theo score desc).
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "content": doc,
            "score": 1.0 - dist,  # cosine distance → cosine similarity
            "metadata": meta,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


if __name__ == "__main__":
    # Test
    query = "hình phạt cho tội tàng trữ ma tuý"
    print(f"Query: {query}\n" + "=" * 60)
    results = semantic_search(query, top_k=5)
    for r in results:
        src = r["metadata"].get("source", "?")
        print(f"[{r['score']:.3f}] ({src}) {r['content'][:100]}...")
