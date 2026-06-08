import os
from openai import OpenAI

def generate_response(query: str, chunks: list, history: list, placeholder) -> str:
    """
    Nhiệm vụ: Gắn ngữ cảnh vào Prompt, gọi OpenAI GPT-4o-mini để sinh ra câu trả lời có trích dẫn.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "⚠️ Vui lòng cấu hình `OPENAI_API_KEY` trong file `.env`"
        placeholder.markdown(msg)
        return msg
        
    client = OpenAI(api_key=api_key)
    
    # 1. Format ngữ cảnh (Context)
    context_text = "\n\n".join([f"Source {i+1}:\n{c['content']}" for i, c in enumerate(chunks)])
    
    # 2. Xây dựng Prompt bằng Tiếng Anh (English System Prompt)
    system_prompt = (
        "You are an expert legal AI assistant specializing in Drug Prevention Laws.\n"
        "Your primary task is to answer the user's [Question] strictly based on the provided [Context] documents.\n\n"
        "CRITICAL RULES:\n"
        "1. NO HALLUCINATION: You must not invent, assume, or infer any information outside the provided context. "
        "If the answer cannot be found in the context, you must clearly state: 'I could not find the relevant information in the provided documents.'\n"
        "2. CITATIONS: You must include precise inline citations for every factual claim you make. "
        "Use the format [1], [2], etc., corresponding to the 'Source N' provided in the context. Do not output a list of references at the end, just the inline citations.\n"
        "3. PROFESSIONALISM: Use clear, formal, and objective language suitable for legal advice.\n"
        "4. OUTPUT LANGUAGE: You MUST ALWAYS generate your final response in Vietnamese (Tiếng Việt), regardless of the language of the question or context."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Đưa lịch sử chat vào để hỗ trợ follow-up questions
    for msg in history[-5:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({
        "role": "user", 
        "content": f"[Context]:\n{context_text}\n\n[Question]: {query}"
    })
    
    try:
        # 3. Gọi API (streaming)
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        full_res = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_res += chunk.choices[0].delta.content
                placeholder.markdown(full_res + "▌")
                
        # 4. Gắn thêm mục danh sách nguồn tài liệu ở cuối
        if chunks:
            full_res += "\n\n---\n**📚 Nguồn tham khảo chi tiết:**\n"
            for i, c in enumerate(chunks):
                source_name = c.get('metadata', {}).get('source', f'Nguồn {i+1}')
                score = c.get('score', 0)
                full_res += f"- **[{i+1}]** {source_name} *(Độ tin cậy: {score:.2f})*\n"
                
        placeholder.markdown(full_res)
        return full_res
        
    except Exception as e:
        err_msg = f"Lỗi gọi LLM: {str(e)}"
        placeholder.markdown(err_msg)
        return err_msg
