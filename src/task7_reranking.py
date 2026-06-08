"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import sys

# Console Windows mặc định dùng cp1252 → không in được ký tự tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

# Cross-encoder reranker chạy LOCAL (không cần API key).
# - bge-reranker-v2-m3: multilingual, tốt cho tiếng Việt, là cặp đôi tự nhiên
#   với embedding bge-m3 ở Task 4. Lần đầu sẽ tải model (~2GB).
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load cross-encoder, cache ở cấp module (tải model 1 lần)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Cross-encoder đưa CẢ (query, document) vào model cùng lúc → đánh giá độ liên
    quan chính xác hơn nhiều so với bi-encoder (dense) vì có attention chéo giữa
    query và doc. Đổi lại chậm hơn → chỉ dùng để rerank top-N candidate.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Chấm điểm từng cặp (query, content).
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    # Gắn điểm mới rồi sort giảm dần, lấy top_k.
    reranked = [
        {**cand, "score": float(score)}
        for cand, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    import numpy as np

    if not candidates:
        return []

    q = np.asarray(query_embedding, dtype=float)
    embs = [np.asarray(c["embedding"], dtype=float) for c in candidates]

    def cosine(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    # Tính sẵn relevance(query, doc) cho mọi candidate.
    relevance = [cosine(q, e) for e in embs]

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            # Độ tương đồng lớn nhất với các doc đã chọn (để phạt trùng lặp).
            max_sim = max(
                (cosine(embs[idx], embs[s]) for s in selected), default=0.0
            )
            mmr = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}   # content -> tổng điểm RRF
    content_map: dict[str, dict] = {}   # content -> dict gốc

    # Mỗi list đóng góp 1/(k+rank) cho từng item. Item xếp hạng cao ở NHIỀU list
    # sẽ có tổng điểm cao → ý tưởng cốt lõi của fusion.
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
    *,
    query_embedding: list[float] | None = None,
    ranked_lists: list[list[dict]] | None = None,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval (cho cross_encoder & mmr)
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking
        query_embedding: Bắt buộc nếu method="mmr"
        ranked_lists: Bắt buộc nếu method="rrf" (nhiều list để fusion)
        lambda_param: Tham số trade-off relevance/diversity cho MMR

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        if query_embedding is None:
            raise ValueError("method='mmr' cần truyền query_embedding")
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)
    elif method == "rrf":
        if ranked_lists is None:
            raise ValueError("method='rrf' cần truyền ranked_lists")
        return rerank_rrf(ranked_lists, top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Dummy data để test nhanh (RRF & MMR không cần tải model).
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}, "embedding": [1.0, 0.0, 0.0]},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}, "embedding": [0.0, 1.0, 0.0]},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}, "embedding": [0.9, 0.1, 0.0]},
    ]

    print("--- RRF (fuse 2 ranked lists) ---")
    list_a = dummy_candidates              # giả lập kết quả semantic
    list_b = list(reversed(dummy_candidates))  # giả lập kết quả lexical
    for r in rerank_rrf([list_a, list_b], top_k=3):
        print(f"[{r['score']:.4f}] {r['content']}")

    print("\n--- MMR (relevance + diversity) ---")
    query_emb = [0.95, 0.05, 0.0]
    for r in rerank_mmr(query_emb, dummy_candidates, top_k=2, lambda_param=0.7):
        print(f"{r['content']}")

    # Cross-encoder cần tải model ~2GB lần đầu → bỏ comment để thử:
    # print("\n--- Cross-encoder ---")
    # for r in rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2):
    #     print(f"[{r['score']:.3f}] {r['content']}")
