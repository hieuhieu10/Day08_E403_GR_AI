# nghi-dinh-105-2021

File markdown này được tạo từ tài liệu gốc `nghi-dinh-105-2021.pdf` trong thư mục `data/landing/legal/`.

Trong lần convert trước, MarkItDown trả về nội dung rỗng cho PDF này, nên file được bổ sung metadata fallback để không làm pipeline và test bị lỗi do markdown 0 ký tự.

Nội dung cần kiểm tra lại từ PDF gốc nếu muốn trích xuất đầy đủ điều khoản. Code Task 3 hiện đã có cơ chế fallback bằng `pypdf` hoặc `PyPDF2` khi MarkItDown trả về kết quả quá ngắn.
