"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank bằng cross-encoder
    4. Nếu top result score < threshold → fallback sang PageIndex (vectorless)
    5. Return top_k results

Mỗi bước được bọc phòng thủ: nếu một retriever lỗi (chưa index, thiếu API key,
offline...) thì pipeline degrade chứ KHÔNG crash — đúng tinh thần "fallback".
"""

import sys
from pathlib import Path

# Cho phép import task5/6/7/8 theo tên trần — khớp đúng quy ước của Task 5/6
# (vốn import `task4_chunking_indexing`). Nhờ vậy chuỗi import giải quyết được dù
# module này được nạp dưới dạng `src.task9_...` hay chạy trực tiếp.
sys.path.insert(0, str(Path(__file__).parent))

# Console Windows mặc định dùng cp1252 → không in được ký tự tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _safe_retrieve(search_fn, query: str, top_k: int, name: str) -> list[dict]:
    """
    Gọi 1 retriever và nuốt lỗi → trả [] nếu thất bại.

    Lý do: thiếu vector store (chưa chạy Task 4), thiếu API key, hay offline
    không được phép làm sập cả pipeline — retriever đó coi như "không đóng góp".
    """
    try:
        return search_fn(query, top_k=top_k)
    except Exception as e:
        print(f"  ⚠ {name} search lỗi → bỏ qua: {e}", file=sys.stderr)
        return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # --- Step 1: Hai nhánh retrieval (lấy dư top_k*2 để merge/rerank có "đất") ---
    dense_results = _safe_retrieve(semantic_search, query, top_k * 2, "semantic")
    sparse_results = _safe_retrieve(lexical_search, query, top_k * 2, "lexical")

    # --- Step 2: Merge bằng RRF (chỉ fuse các list thực sự có kết quả) ---
    ranked_lists = [lst for lst in (dense_results, sparse_results) if lst]
    merged = rerank_rrf(ranked_lists, top_k=top_k * 2) if ranked_lists else []
    for item in merged:
        item["source"] = "hybrid"

    # --- Step 3: Rerank (cross-encoder). Lỗi model → degrade về thứ tự RRF ---
    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item["source"] = "hybrid"  # rerank giữ nguyên dict nhưng đảm bảo nhãn
        except Exception as e:
            print(f"  ⚠ Rerank lỗi → dùng thứ tự RRF: {e}", file=sys.stderr)
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # --- Step 4: Threshold check → fallback PageIndex (vectorless) ---
    best_score = final_results[0]["score"] if final_results else 0.0
    if best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback → PageIndex",
            file=sys.stderr,
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
            print("  ⚠ PageIndex không có kết quả → giữ hybrid", file=sys.stderr)
        except Exception as e:
            # PageIndex chưa sẵn sàng (thiếu key/doc/offline) → giữ hybrid hiện có.
            print(f"  ⚠ PageIndex fallback không khả dụng → giữ hybrid: {e}", file=sys.stderr)

    # --- Step 5 ---
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
