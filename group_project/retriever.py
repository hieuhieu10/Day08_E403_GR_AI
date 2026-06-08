import streamlit as st
import chromadb
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Sử dụng st.cache_resource để tránh việc load model/db nhiều lần gây giật lag
@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_collection():
    db_path = str(Path(__file__).parent / "chroma_db")
    try:
        client = chromadb.PersistentClient(path=db_path)
        return client.get_collection(name="rag_collection")
    except Exception:
        return None

@st.cache_resource
def get_bm25_data():
    bm25_path = Path(__file__).parent / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            return pickle.load(f)
    return None

def retrieve_dense(query: str, top_k: int = 15) -> list:
    """Semantic Search bằng Vector Similarity (ChromaDB)"""
    model = get_model()
    collection = get_chroma_collection()
    if not collection:
        return []
        
    query_embedding = model.encode(query).tolist()
    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "score": 1 - results['distances'][0][i], # Chuyển distance thành score
                    "metadata": results['metadatas'][0][i],
                    "type": "dense"
                })
        return chunks
    except Exception as e:
        print(f"Lỗi Dense Retrieval: {e}")
        return []

def retrieve_sparse(query: str, top_k: int = 15) -> list:
    """Lexical Search bằng thuật toán BM25Okapi"""
    bm25_data = get_bm25_data()
    if not bm25_data:
        return []
        
    bm25 = bm25_data["bm25"]
    tokenized_query = query.lower().split()
    
    # Lấy điểm số BM25 của toàn bộ văn bản
    scores = bm25.get_scores(tokenized_query)
    
    # Lấy top K phần tử có điểm cao nhất
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    chunks = []
    for idx in top_indices:
        if scores[idx] > 0: # Chỉ lấy các kết quả có điểm match > 0
            chunks.append({
                "id": bm25_data["ids"][idx],
                "content": bm25_data["chunks"][idx],
                "score": float(scores[idx]),
                "metadata": bm25_data["metadatas"][idx],
                "type": "sparse"
            })
    return chunks