"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
import sys

from task4_chunking_indexing import chunk_documents, load_documents

# Console Windows mặc định dùng cp1252 → không in được ký tự tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

# Corpus = đúng các chunk của Task 4 (tái dùng load + chunk) → khớp với Task 5,
# tiện kết hợp hybrid ở Task 9. BM25 chỉ cần text, không cần embedding.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# Cache index ở cấp module → build 1 lần, tái dùng cho mọi truy vấn.
_bm25 = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản cho BM25: lowercase + tách theo từ (giữ chữ số & tiếng Việt).

    Ghi chú demo (+bonus): với tiếng Việt, dùng underthesea.word_tokenize sẽ tách
    được từ ghép ("tàng trữ", "ma túy") tốt hơn split() → tăng độ chính xác BM25.
    Ở đây dùng regex \\w (có Unicode) cho nhẹ và không thêm dependency nặng.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index (k1=1.5, b=0.75 mặc định).
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_index():
    """Lazy-load corpus + build BM25 index 1 lần."""
    global _bm25
    if _bm25 is None:
        documents = load_documents()
        CORPUS[:] = chunk_documents(documents)  # điền CORPUS in-place
        _bm25 = build_bm25_index(CORPUS)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    import numpy as np

    bm25 = _get_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k index có điểm cao nhất, sort giảm dần.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # bỏ chunk không match từ khóa nào
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    query = "Điều 248 tàng trữ trái phép chất ma tuý"
    print(f"Query: {query}\n" + "=" * 60)
    results = lexical_search(query, top_k=5)
    for r in results:
        src = r["metadata"].get("source", "?")
        print(f"[{r['score']:.3f}] ({src}) {r['content'][:100]}...")
