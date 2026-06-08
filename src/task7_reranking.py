"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional

import numpy as np


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vector embedding."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _token_overlap_score(query: str, content: str) -> float:
    """Tính điểm overlap đơn giản giữa query và nội dung."""
    query_tokens = set(query.lower().split())
    content_tokens = set(content.lower().split())
    if not query_tokens or not content_tokens:
        return 0.0
    return float(len(query_tokens & content_tokens) / len(query_tokens))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Chấm lại và sắp xếp candidates theo mức độ liên quan với query."""
    if not candidates:
        return []

    reranked = []
    for cand in candidates:
        content = cand.get("content", "")
        base_score = float(cand.get("score", 0.0))
        overlap = _token_overlap_score(query, content)
        # Cross-encoder thật thường chấm query+doc trực tiếp; ở đây dùng
        # một heuristic local để pipeline chạy được ngay.
        rerank_score = 0.7 * overlap + 0.3 * base_score
        reranked.append({**cand, "score": rerank_score})

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Chọn các candidates vừa liên quan vừa ít trùng lặp bằng MMR."""
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            cand_embedding = candidates[idx].get("embedding")
            if cand_embedding is None:
                continue

            relevance = _cosine_sim(query_embedding, cand_embedding)

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_embedding = candidates[sel_idx].get("embedding")
                if sel_embedding is None:
                    continue
                sim = _cosine_sim(cand_embedding, sel_embedding)
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Gộp nhiều ranked lists thành một danh sách cuối bằng RRF."""
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
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
) -> list[dict]:
    """Chọn và gọi phương pháp reranking phù hợp theo `method`."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
