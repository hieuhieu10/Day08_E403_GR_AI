"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
PAGEINDEX_DIR = Path(__file__).parent.parent / "data" / "pageindex"
PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = PAGEINDEX_DIR / "uploaded_documents.json"


def _get_client():
    """Tạo PageIndex client từ API key đã cấu hình."""
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong .env")

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_uploaded_manifest() -> list[dict]:
    """Đọc manifest các tài liệu đã upload nếu có."""
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_uploaded_manifest(items: list[dict]) -> None:
    """Lưu manifest doc_id và metadata để query lại sau."""
    MANIFEST_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _get_client()
    uploaded: list[dict] = []

    # PageIndex SDK hiện tại chỉ hỗ trợ PDF. Vì vậy ta upload các file PDF gốc
    # trong data/landing/legal/ thay vì markdown standardized.
    pdf_files = list((LANDING_DIR / "legal").rglob("*.pdf"))
    if not pdf_files:
        print("Không tìm thấy PDF nào trong data/landing/legal/")
        return

    for pdf_file in pdf_files:
        print(f"Uploading: {pdf_file.name}")

        submit_result = client.submit_document(str(pdf_file))
        doc_id = submit_result.get("doc_id") if isinstance(submit_result, dict) else submit_result
        if not doc_id:
            raise RuntimeError(f"Không lấy được doc_id sau khi upload {pdf_file.name}")

        # Chờ tài liệu sẵn sàng cho retrieval.
        for _ in range(60):
            try:
                if hasattr(client, "is_retrieval_ready"):
                    ready = client.is_retrieval_ready(doc_id)
                    if ready:
                        break
                else:
                    doc_meta = client.get_document(doc_id)
                    status = str(doc_meta).lower()
                    if "ready" in status or "complete" in status:
                        break
            except Exception:
                pass
            time.sleep(5)

        uploaded.append(
            {
                "doc_id": doc_id,
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "type": pdf_file.parent.name,
            }
        )
        print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")

    _save_uploaded_manifest(uploaded)
    print(f"\n✓ Saved manifest: {MANIFEST_PATH}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    client = _get_client()
    uploaded = _load_uploaded_manifest()
    if not uploaded:
        return []

    query_tokens = set(query.lower().split())
    scored_results: list[dict] = []

    for item in uploaded:
        doc_id = item["doc_id"]
        try:
            tree_payload = client.get_document_structure(doc_id)
            tree = json.loads(tree_payload) if isinstance(tree_payload, str) else tree_payload
        except Exception:
            continue

        def walk_nodes(nodes):
            for node in nodes or []:
                yield node
                yield from walk_nodes(node.get("nodes", []))

        for node in walk_nodes(tree):
            title = str(node.get("title", ""))
            summary = str(node.get("summary", "") or node.get("prefix_summary", ""))
            node_tokens = set(f"{title} {summary}".lower().split())
            overlap = len(query_tokens & node_tokens)
            if overlap == 0:
                continue

            score = overlap / max(len(query_tokens), 1)
            page_index = node.get("page_index")
            content = summary

            if page_index is not None:
                try:
                    page_content = client.get_page_content(doc_id, str(page_index))
                    if page_content:
                        content = page_content
                except Exception:
                    pass

            scored_results.append(
                {
                    "content": content,
                    "score": float(score),
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": item["filename"],
                        "type": item["type"],
                        "title": title,
                        "page_index": page_index,
                    },
                    "source": "pageindex",
                }
            )

    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Loại trùng theo content, giữ kết quả tốt nhất.
    seen = set()
    results: list[dict] = []
    for item in scored_results:
        key = item["content"][:200]
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
