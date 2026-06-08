"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(__file__).parent.parent / "data" / "vectorstore" / "task4_index.json"
EMBEDDING_MODEL = "BAAI/bge-m3"


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
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy index tại {INDEX_PATH}. Hãy chạy Task 4 trước."
        )

    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        return []

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    chunk_embeddings = np.asarray(
        [chunk["embedding"] for chunk in chunks], dtype=np.float32
    )

    # Vì embedding đã được normalize, cosine similarity = dot product.
    scores = chunk_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        chunk = chunks[int(idx)]
        results.append(
            {
                "content": chunk["content"],
                "score": score,
                "metadata": chunk.get("metadata", {}),
            }
        )

    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
