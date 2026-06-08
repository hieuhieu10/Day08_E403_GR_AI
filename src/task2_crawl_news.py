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
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vietnamnet.vn/ca-si-miu-le-duong-tinh-voi-ma-tuy-bi-bat-giu-o-hai-phong-2514728.html",
    "https://tuoitre.vn/ca-si-long-nhat-va-son-ngoc-minh-mua-ma-tuy-tu-dau-20260522101239371.htm",
    "https://baovanhoa.vn/giai-tri/ma-tuy-va-nhung-cu-nga-ngua-cua-showbiz-viet-230477.html",
    "https://kenh14.vn/loat-sao-viet-vuong-vong-lao-ly-vi-ma-tuy-nguoi-mat-su-nghiep-ke-linh-an-hang-chuc-nam-tu-215260511192134183.chn",
    "https://nld.com.vn/tu-on-ao-miu-le-nhung-vu-be-boi-ma-tuy-gay-soc-showbiz-viet-196260511232148677.htm",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") or getattr(result, "title", None) or "Unknown"
    content_markdown = (
        getattr(result, "markdown", None)
        or getattr(result, "cleaned_markdown", None)
        or getattr(result, "text", None)
        or ""
    )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
        "source_metadata": {
            "site_name": metadata.get("site_name"),
            "description": metadata.get("description"),
        },
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
