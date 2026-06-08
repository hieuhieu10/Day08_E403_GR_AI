"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Console Windows mặc định dùng cp1252 → không in được ký tự ✓/✗/tiếng Việt.
# Ép stdout sang UTF-8 để tránh UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://dantri.com.vn/phap-luat/khoi-to-bat-giam-nguoi-mau-andrea-aybar-ca-si-chi-dan-20241114115057035.htm",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
]


async def crawl_article(crawler, url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Args:
        crawler: instance AsyncWebCrawler đang mở (tái sử dụng cho mọi URL).
        url: địa chỉ bài báo cần crawl.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }

    Raises:
        RuntimeError: khi crawl thất bại (trang lỗi, bị chặn, timeout...).
    """
    result = await crawler.arun(url=url)

    if not result.success:
        raise RuntimeError(result.error_message or "Crawl thất bại (không rõ lý do)")

    metadata = result.metadata or {}
    # crawl4ai trả về markdown có thể là str hoặc object (MarkdownGenerationResult)
    markdown = getattr(result.markdown, "raw_markdown", result.markdown) or ""

    return {
        "url": url,
        "title": metadata.get("title", "Unknown"),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    from crawl4ai import AsyncWebCrawler

    setup_directory()

    success_count = 0
    failed: list[tuple[str, str]] = []

    # Mở crawler 1 lần và tái sử dụng cho toàn bộ URL (nhanh & tiết kiệm tài nguyên)
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            try:
                article = await crawl_article(crawler, url)

                # Lưu file JSON
                filename = f"article_{i:02d}.json"
                filepath = DATA_DIR / filename
                filepath.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                success_count += 1
                print(f"  ✓ Saved: {filepath}")
            except Exception as exc:  # noqa: BLE001 - log mọi lỗi, không dừng cả batch
                failed.append((url, str(exc)))
                print(f"  ✗ Lỗi: {exc}")

    # Tổng kết
    print("\n" + "=" * 50)
    print(f"Hoàn tất: {success_count}/{len(ARTICLE_URLS)} bài crawl thành công.")
    if failed:
        print(f"Thất bại ({len(failed)}):")
        for url, err in failed:
            print(f"  - {url}\n    → {err}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
