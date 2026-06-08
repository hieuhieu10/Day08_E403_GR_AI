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
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def extract_pdf_text_fallback(filepath: Path) -> str:
    """Extract text từ PDF nếu MarkItDown trả về nội dung rỗng."""
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        reader = PdfReader(str(filepath))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"## Page {i}\n\n{text.strip()}")
        return "\n\n".join(pages)
    except Exception as exc:
        return (
            f"# {filepath.stem}\n\n"
            f"Không thể trích xuất nội dung PDF bằng MarkItDown hoặc fallback PDF reader.\n\n"
            f"File gốc: `{filepath.name}`\n\n"
            f"Lỗi fallback: `{exc}`\n\n"
            "Ghi chú: file này vẫn được ghi nhận trong pipeline để tránh tạo markdown rỗng. "
            "Nếu cần nội dung đầy đủ, hãy kiểm tra lại PDF gốc hoặc dùng OCR cho file scan."
        )


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text_content = result.text_content or ""

            if filepath.suffix.lower() == ".pdf" and len(text_content.strip()) < 200:
                print("  MarkItDown output quá ngắn, thử PDF fallback...")
                text_content = extract_pdf_text_fallback(filepath)

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text_content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date_crawled = data.get("date_crawled", "N/A")
            content_markdown = data.get("content_markdown", "") or ""

            header = f"# {title}\n\n"
            header += f"**Source:** {url}\n"
            header += f"**Crawled:** {date_crawled}\n\n"
            header += "---\n\n"

            output_path.write_text(header + content_markdown, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
