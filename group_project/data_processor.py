import os
import pickle
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def load_documents(data_dir: Path):
    docs = []
    # Quét tất cả file .md và .txt trong thư mục data/
    for ext in ["*.md", "*.txt"]:
        for file_path in data_dir.rglob(ext):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    docs.append({"content": f.read(), "source": file_path.name})
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")
    return docs

def process_and_index():
    print("Bắt đầu quá trình xử lý dữ liệu và tạo Database...")
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "standardized"
    
    if not data_dir.exists():
        print(f"Thư mục dữ liệu không tồn tại: {data_dir}")
        return
        
    documents = load_documents(data_dir)
    if not documents:
        print("⚠ Không tìm thấy tài liệu nào trong thư mục data/standardized.")
        return
        
    print(f"Đã tải {len(documents)} tài liệu. Bắt đầu Chunking...")
    
    # 1. Thuật toán Chunking (Langchain)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    metadatas = []
    ids = []
    
    chunk_id = 0
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append(split)
            metadatas.append({"source": doc["source"]})
            ids.append(f"chunk_{chunk_id}")
            chunk_id += 1
            
    print(f"Đã tạo {len(chunks)} chunks. Bắt đầu Embedding và lưu ChromaDB...")
    
    # 2. Dense Vector Indexing (ChromaDB)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks).tolist()
    
    db_path = Path(__file__).parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Xóa collection cũ nếu có để tạo lại từ đầu cho sạch
    try:
        client.delete_collection("rag_collection")
    except:
        pass
        
    collection = client.create_collection(name="rag_collection")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print("Đã lưu Dense Vector Database thành công vào ChromaDB!")
    
    # 3. Sparse Lexical Indexing (BM25Okapi)
    print("Bắt đầu xây dựng BM25 Index cho Lexical Search...")
    tokenized_corpus = [doc.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25": bm25,
        "chunks": chunks,
        "metadatas": metadatas,
        "ids": ids
    }
    
    bm25_path = Path(__file__).parent / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
        
    print("Đã lưu Lexical Index (BM25) thành công!")
    print("Hoàn tất quá trình xử lý dữ liệu! Sẵn sàng cho Chatbot.")

if __name__ == "__main__":
    process_and_index()