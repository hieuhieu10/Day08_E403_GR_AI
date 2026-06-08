import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load môi trường từ thư mục gốc
load_dotenv(Path(__file__).parent.parent / '.env')
# Đưa thư mục group_project vào sys path để dễ import các module con
sys.path.insert(0, str(Path(__file__).parent))

from retriever import retrieve_dense, retrieve_sparse
from reranker import rrf_fusion, reorder_lost_in_the_middle
from generator import generate_response

st.set_page_config(page_title="RAG Chatbot Professional", page_icon="⚖️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4D96FF;'>⚖️ RAG Chatbot - Luật Ma Túy</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Kiến trúc Hybrid Search + RRF + Lost In The Middle</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Khôi phục lịch sử chat trên giao diện
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn..."):
    # Lưu câu hỏi vào session
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Đang quét tài liệu bằng Hybrid Search và Reranking..."):
            # BƯỚC 1: Tìm kiếm tài liệu bằng 2 phương pháp (Diệu Linh)
            dense_chunks = retrieve_dense(prompt, top_k=15)
            sparse_chunks = retrieve_sparse(prompt, top_k=15)
            
            # BƯỚC 2: Reranking & Tinh chỉnh kết quả (Minh Hiếu)
            # Gộp điểm bằng thuật toán Reciprocal Rank Fusion
            fused_chunks = rrf_fusion(dense_chunks, sparse_chunks, top_k=5)
            # Sắp xếp lại theo mô hình Lost in the Middle
            optimized_chunks = reorder_lost_in_the_middle(fused_chunks)
            
        # BƯỚC 3: Sinh văn bản thông minh bằng LLM (Minh Quang)
        full_response = generate_response(
            query=prompt, 
            chunks=optimized_chunks, 
            history=st.session_state.messages[:-1], 
            placeholder=response_placeholder
        )
        
        # Lưu câu trả lời của AI vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": full_response})
