"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import sys
from pathlib import Path

from markitdown import MarkItDown

# Console Windows mặc định dùng cp1252 → không in được ký tự ✓/✗/tiếng Việt.
# Ép stdout sang UTF-8 để tránh UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> tuple[int, int]:
    """
    Convert PDF/DOCX files trong data/landing/legal/ sang markdown.

    Returns:
        (số file thành công, tổng số file đã thử).
    """
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"

    if not legal_dir.exists():
        print(f"  ⚠ Bỏ qua: chưa có thư mục {legal_dir}")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    md = MarkItDown()

    success = total = 0
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        total += 1
        print(f"Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            success += 1
            print(f"  ✓ Saved: {output_path}")
        except Exception as exc:  # noqa: BLE001 - log lỗi, tiếp tục file khác
            print(f"  ✗ Lỗi: {exc}")

    return success, total


def convert_news_articles() -> tuple[int, int]:
    """
    Convert JSON crawled articles trong data/landing/news/ sang markdown.

    Returns:
        (số file thành công, tổng số file đã thử).
    """
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"

    if not news_dir.exists():
        print(f"  ⚠ Bỏ qua: chưa có thư mục {news_dir}")
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    success = total = 0
    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() != ".json":
            continue
        total += 1
        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Thêm metadata header để giữ nguồn gốc bài báo
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content, encoding="utf-8")
            success += 1
            print(f"  ✓ Saved: {output_path}")
        except Exception as exc:  # noqa: BLE001 - log lỗi, tiếp tục file khác
            print(f"  ✗ Lỗi: {exc}")

    return success, total


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_ok, legal_total = convert_legal_docs()

    print("\n--- News Articles ---")
    news_ok, news_total = convert_news_articles()

    print("\n" + "=" * 50)
    print(f"Legal: {legal_ok}/{legal_total} | News: {news_ok}/{news_total}")
    print("✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
