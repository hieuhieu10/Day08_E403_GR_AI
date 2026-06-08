import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Console Windows mặc định dùng cp1252 → không in được ký tự tiếng Việt.
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

# Upload bản PDF GỐC (data/landing/legal/) chứ không phải markdown đã convert:
# PageIndex chỉ nhận PDF và tự OCR + dựng cây mục lục từ layout gốc. Các văn bản
# luật là PDF có cấu trúc Điều/Khoản rõ ràng → hợp nhất với cách duyệt cây.
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Lưu lại mapping {filename: doc_id} sau khi upload để search KHÔNG phải upload
# lại mỗi lần chạy (upload + dựng cây tốn thời gian và quota).
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

# Trạng thái coi như "đã xong" khi poll retrieval / xử lý document.
_DONE_STATES = {"completed", "complete", "done", "success", "finished", "ready"}
_FAIL_STATES = {"failed", "error", "cancelled", "canceled"}

# Các tên trường text hay gặp trong 1 node trả về (schema có thể đổi theo version).
_TEXT_KEYS = (
    "relevant_contents", "relevant_content", "contents", "content",
    "text", "page_content",
)

_client = None


def _get_client():
    """Lazy-init PageIndexClient (import trong hàm để module luôn import được
    kể cả khi chưa cài pageindex / chưa có API key)."""
    global _client
    if _client is None:
        if not PAGEINDEX_API_KEY:
            raise RuntimeError(
                "Thiếu PAGEINDEX_API_KEY. Đăng ký tại https://pageindex.ai/ "
                "rồi set trong file .env."
            )
        from pageindex import PageIndexClient
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


# =============================================================================
# UPLOAD
# =============================================================================

def upload_documents(wait: bool = True, timeout: int = 600) -> dict:
    """
    Upload toàn bộ PDF trong data/landing/legal/ lên PageIndex và lưu doc_id.

    Args:
        wait: Nếu True, chờ tới khi document sẵn sàng cho retrieval.
        timeout: Thời gian chờ tối đa cho mỗi document (giây).

    Returns:
        dict {filename: doc_id}. Đồng thời ghi ra DOC_IDS_FILE để tái dùng.
    """
    client = _get_client()

    pdfs = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Không tìm thấy PDF nào trong {LEGAL_DIR}. "
            "Hãy hoàn thành Task 1 (tải văn bản luật) trước."
        )

    doc_ids = _load_doc_ids()  # giữ lại doc_id cũ, chỉ upload file mới
    for pdf in pdfs:
        if pdf.name in doc_ids:
            print(f"  ⏭ Bỏ qua (đã upload): {pdf.name} -> {doc_ids[pdf.name]}")
            continue
        resp = client.submit_document(file_path=str(pdf))
        doc_id = resp["doc_id"]
        doc_ids[pdf.name] = doc_id
        print(f"  ✓ Uploaded: {pdf.name} -> {doc_id}")

    _save_doc_ids(doc_ids)

    if wait:
        for fname, doc_id in doc_ids.items():
            print(f"  ⏳ Chờ xử lý: {fname} ...", end="", flush=True)
            ready = _wait_until(lambda: client.is_retrieval_ready(doc_id), timeout)
            print(" sẵn sàng" if ready else " (timeout, có thể vẫn đang xử lý)")

    return doc_ids


def _load_doc_ids() -> dict:
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_doc_ids(doc_ids: dict) -> None:
    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _wait_until(predicate, timeout: int, interval: float = 3.0) -> bool:
    """Poll predicate() tới khi True hoặc hết timeout. Trả về kết quả cuối cùng."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass  # đang xử lý → API có thể tạm lỗi, cứ thử lại
        time.sleep(interval)
    return False


# =============================================================================
# RETRIEVAL
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex. Truy vấn trên TẤT CẢ document đã upload,
    gộp kết quả và trả về top_k node liên quan nhất.

    Dùng làm fallback khi hybrid search không cho kết quả tốt (xem Task 9).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,        # {'source': filename, 'node_id': ..., 'title': ...}
            'source': 'pageindex'    # Đánh dấu nguồn retrieval (Task 9 phân biệt)
        }
        Sorted by score descending.
    """
    client = _get_client()

    doc_ids = _load_doc_ids()
    if not doc_ids:
        raise RuntimeError(
            "Chưa có document nào trên PageIndex. Chạy upload_documents() trước "
            "(hoặc `python -m src.task8_pageindex_vectorless`)."
        )

    results: list[dict] = []
    for fname, doc_id in doc_ids.items():
        # 1) Gửi truy vấn → nhận retrieval_id.
        retrieval_id = client.submit_query(doc_id=doc_id, query=query)["retrieval_id"]

        # 2) Poll tới khi retrieval hoàn tất.
        retrieval = _poll_retrieval(client, retrieval_id)
        if retrieval is None:
            continue

        # 3) Trích các node liên quan và chuẩn hoá về format chung.
        nodes = _find_nodes(retrieval)
        for rank, node in enumerate(nodes):
            content = _node_text(node)
            if not content:
                continue
            results.append({
                "content": content,
                "score": _node_score(node, rank),
                "metadata": {
                    "source": fname,
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                },
                "source": "pageindex",
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _poll_retrieval(client, retrieval_id: str, timeout: int = 120) -> dict | None:
    """Poll get_retrieval tới khi 'completed'. Trả về dict kết quả, hoặc None nếu lỗi/timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get_retrieval(retrieval_id)
        status = str(res.get("status", "")).lower()
        if status in _FAIL_STATES:
            return None
        # Có node để đọc, hoặc API báo xong → trả luôn.
        if status in _DONE_STATES or _find_nodes(res):
            return res
        time.sleep(2.0)
    return None


def _find_nodes(obj) -> list[dict]:
    """
    Tìm list node liên quan đầu tiên trong response của get_retrieval.

    Viết kiểu phòng thủ (đệ quy) để không phụ thuộc cứng vào tên key — schema
    của PageIndex có thể đổi giữa các version (đã gặp 'retrieval', 'sources',
    'retrieved_nodes', 'nodes'...). Coi 1 list là "list node" nếu có phần tử dict
    chứa text đọc được.
    """
    if isinstance(obj, list):
        node_dicts = [x for x in obj if isinstance(x, dict)]
        if node_dicts and any(_node_text(x) for x in node_dicts):
            return node_dicts
        for item in obj:
            found = _find_nodes(item)
            if found:
                return found
        return []
    if isinstance(obj, dict):
        for key in ("sources", "retrieved_nodes", "nodes", "retrieval", "results"):
            if key in obj:
                found = _find_nodes(obj[key])
                if found:
                    return found
        for value in obj.values():
            found = _find_nodes(value)
            if found:
                return found
    return []


def _node_text(node: dict) -> str:
    """Lấy text của 1 node, chịu được nhiều tên trường khác nhau."""
    for key in _TEXT_KEYS:
        val = node.get(key)
        if isinstance(val, list):
            val = "\n".join(str(x) for x in val)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _node_score(node: dict, rank: int) -> float:
    """
    Điểm liên quan của node. PageIndex thường trả node theo thứ tự liên quan giảm
    dần nhưng KHÔNG kèm điểm số → ta suy ra điểm theo thứ hạng để Task 9 còn so
    sánh/merge được với semantic & lexical. Nếu API có trả 'score'/'relevance'
    thì dùng luôn.
    """
    for key in ("score", "relevance", "relevance_score"):
        if isinstance(node.get(key), (int, float)):
            return float(node[key])
    return round(1.0 / (rank + 1), 4)  # 1.0, 0.5, 0.33, ... theo thứ hạng


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        if not results:
            print("  (không có kết quả — document có thể vẫn đang xử lý)")
        for r in results:
            src = r["metadata"].get("source", "?")
            print(f"[{r['score']:.3f}] ({src}) {r['content'][:100]}...")
