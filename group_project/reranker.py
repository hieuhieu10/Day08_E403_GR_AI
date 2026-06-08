def rrf_fusion(dense_results: list, sparse_results: list, top_k: int = 5, k: int = 60) -> list:
    """
    Thuật toán Reciprocal Rank Fusion (RRF)
    Kết hợp kết quả từ Semantic Search và Keyword Search để tạo ra Hybrid Search công bằng.
    Công thức RRF Score = 1 / (k + rank)
    """
    rrf_scores = {}
    chunk_map = {}

    # 1. Tính điểm RRF cho các kết quả Dense
    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[doc_id] = item

    # 2. Tính điểm RRF cho các kết quả Sparse (BM25)
    for rank, item in enumerate(sparse_results):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[doc_id] = item

    # 3. Xếp hạng lại toàn bộ theo tổng điểm RRF
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_id, score in sorted_items[:top_k]:
        chunk = chunk_map[doc_id].copy()
        chunk["rrf_score"] = score
        fused_results.append(chunk)

    return fused_results


def reorder_lost_in_the_middle(chunks: list) -> list:
    """
    Thuật toán Lost in the Middle Reordering:
    Mô hình LLM thường chú ý nhiều nhất ở phần ĐẦU và CUỐI của đoạn văn bản dài.
    Do đó, ta xếp các chunks có điểm cao nhất (quan trọng nhất) ở vị trí đầu và cuối,
    những chunk kém hơn sẽ được đẩy vào giữa context.
    """
    if not chunks:
        return []

    reordered = []
    # Thuật toán: Phần tử chẵn nhét vào đầu, phần tử lẻ nhét vào cuối
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered.insert(0, chunk)
        else:
            reordered.append(chunk)

    return reordered