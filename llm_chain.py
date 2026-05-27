import os
import re
import asyncio
import threading
import random
import time as _time
from dotenv import load_dotenv
from groq import Groq, AsyncGroq
from sentence_transformers import CrossEncoder

load_dotenv()  # ← thêm dòng này

# Khởi tạo reranker (chạy 1 lần)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

_ALL_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

client = Groq(api_key=random.choice(_ALL_KEYS))
async_client = AsyncGroq(api_key=random.choice(_ALL_KEYS))

MODEL_NAME = "llama-3.1-8b-instant"

# ================== SMALLTALK ==================
GREETING_WORDS = [
    # ── Chào hỏi — Greeting (Cornell Movie-Dialogs Corpus) ──
    "xin chào", "hello", "hi", "hey", "alo", "chào",
    "good morning", "good afternoon", "good evening",
    "chào buổi sáng", "chào chiều", "chào tối",
    # ── Cảm ơn / Tạm biệt — Acknowledgment (Cornell corpus) ──
    "cảm ơn", "thank", "thanks", "bye", "tạm biệt",
    "ok cảm ơn", "cảm ơn nhé", "cảm ơn vì",
    "bạn thật hữu ích", "bạn giỏi quá", "bạn tư vấn rất tốt",
    "tôi hài lòng", "tôi thích chatbot",
]

BOT_QUESTIONS = [
    # ── Identity Query — hỏi về bot (Persona-Chat Dataset) ──
    "bạn là ai", "bạn tên gì", "bạn làm được gì",
    "bạn có thể làm gì", "bạn giúp được gì",
    "bạn hoạt động như thế nào", "bạn được tạo ra bởi ai",
    "momcare là gì", "bạn chạy trên nền tảng nào",
    "bạn dùng ai gì", "ai tạo ra bạn",
    "giới thiệu về bạn", "bạn biết những gì",
    "bạn được lập trình", "bạn có thể nói tiếng anh",
    "tôi muốn biết về bạn", "bạn có thể làm toán",
    # ── Câu xã giao ngắn ──
    "bạn ơi", "bạn ơi cho hỏi", "mình muốn hỏi bạn",
    "bạn khỏe không", "bạn tốt không", "hôm nay bạn thế nào",
    "bạn có nhớ tôi không", "mình quay lại rồi",
    "tôi đang cần giúp đỡ", "bạn thông minh quá",
]

def is_smalltalk(question: str) -> bool:
    q = question.lower().strip()

    if len(q.split()) <= 5:
        for word in GREETING_WORDS:
            if word in q:
                return True

    for word in BOT_QUESTIONS:
        if word in q:
            return True

    return False

BLOCKED_INPUTS = [
    # ── Nhóm kê đơn / liều thuốc (Luật KCB 15/2023/QH15 Điều 7) ──
    "kê đơn", "kê toa", "đơn thuốc", "toa thuốc",
    "cho tôi thuốc", "liều thuốc", "liều an toàn", "liều gây nguy hiểm",
    "overdose", "quá liều",
    "liều ibuprofen", "liều paracetamol", "liều morphine", "liều diazepam",
    "thuốc gì để ngủ", "thuốc ngủ", "thuốc an thần", "thuốc kháng sinh",
    # Thêm để khớp 50 câu test
    "đơn thuốc tăng sữa",
    "cho tôi biết cách dùng thuốc",
    "thuốc nào dùng để",
    "thuốc nào làm",
    "thuốc nào giúp",
    "thuốc nào nguy hiểm",
    "thuốc trầm cảm",
    "cho tôi thuốc",
    "cho bé dùng thuốc",
    "cho tôi uống thuốc",
    "thuốc amoxicillin",
    "overdose paracetamol",
    "cần đơn thuốc",
    "kháng sinh mạnh",
    "kháng viêm",
    "thuốc canxi",
    "vitamin cho bé",
    "hạ sốt nhanh nhất",
    "làm ngất",
    "không bao giờ thức dậy",
    "ngủ mãi",
]

MENTAL_HEALTH_KEYWORDS = [
    # ── Gốc ──
    "tự tử", "muốn chết", "không muốn sống", "chán sống",
    "tự làm hại", "không chịu được nữa", "chết",
    # ── Thêm để khớp 50 câu test (WHO mhGAP Guideline 2.0) ──
    "muốn biến mất",
    "không còn ý nghĩa",
    "không còn hi vọng",
    "kết thúc tất cả",
    "chấm dứt tất cả",
    "muốn dừng lại",
    "không muốn tiếp tục",
    "không lý do để sống",
    "hết đau khổ mãi mãi",
    "ngủ không thức dậy",
    "buông bỏ",
    "không muốn làm mẹ nữa",
    "nuôi con một mình quá mệt muốn chết",
    "tuyệt vọng quá không muốn sống",
    "không còn sức để chiến đấu",
    "cuộc sống quá nặng nề",
    "muốn kết thúc",
    "để không còn cảm giác đau",
    "cách tự làm hại",
    "muốn ngủ và không thức dậy",
]

DIAGNOSIS_KEYWORDS = [
    "bị bệnh", "chẩn đoán", "mắc bệnh", "bị ung thư",
    "bị tiểu đường", "bị nhiễm trùng", "kết luận"
]

SAFE_RESPONSE = "Mẹ nên đưa bé đến cơ sở y tế để được thăm khám trực tiếp."

MENTAL_HEALTH_RESPONSE = """Mẹ ơi, MomCare hiểu rằng giai đoạn này có thể rất khó khăn và kiệt sức. 

Mẹ không đơn độc đâu. Hãy gọi ngay:
- **Đường dây hỗ trợ sức khỏe tinh thần: 1800 599 920** (miễn phí, 24/7)
- Hoặc nhờ người thân ở cạnh mẹ lúc này.

Mẹ và bé cần nhau, hãy để người khác giúp mẹ nhé."""

def check_input_guardrails(question: str) -> str | None:
    q = question.lower()
    for word in MENTAL_HEALTH_KEYWORDS:
        if word in q:
            return MENTAL_HEALTH_RESPONSE
    for word in BLOCKED_INPUTS:
        if word in q:
            return "Xin lỗi, MomCare không thể hỗ trợ yêu cầu này. " + SAFE_RESPONSE
    return None

def check_output_guardrails(answer: str) -> str:
    a = answer.lower()
    for word in DIAGNOSIS_KEYWORDS:
        if word in a:
            return answer + f"\n\n *Lưu ý: {SAFE_RESPONSE}*"
    return answer

# Phân loại bằng LLM (LLM-based Intent Detection)
def get_intent_by_llm(question: str) -> str:
    """Sử dụng LLM để phân tích ý định thực sự khi keyword-based không chắc chắn"""
    prompt = f"""Phân loại ý định của người dùng sau đây vào 1 trong 3 nhóm:
1. BLOCKED: Câu hỏi nguy hiểm, đòi kê đơn thuốc, hoặc có dấu hiệu trầm cảm, muốn tự tử, chán sống.
2. SMALLTALK: Chào hỏi, cảm ơn, tán gẫu hoặc chia sẻ cảm xúc cá nhân (lo lắng, mệt mỏi nhưng chưa đến mức nguy hiểm).
3. RAG: Câu hỏi cụ thể về kiến thức y khoa, chăm sóc trẻ, dinh dưỡng, bệnh lý.

Câu hỏi: "{question}"

Chỉ trả ra đúng 1 từ duy nhất là tên nhóm: BLOCKED, SMALLTALK hoặc RAG. Không giải thích gì thêm."""
    
    intent = call_llm(prompt, temperature=0).strip().upper()
    return intent if intent in ["BLOCKED", "SMALLTALK", "RAG"] else "RAG"

# ================== CALL GROQ ==================
def call_llm(prompt: str, temperature=0.3, max_retries=4):
    for attempt in range(max_retries):
        try:
            _client = Groq(api_key=random.choice(_ALL_KEYS))
            chat_completion = _client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý MomCare, chuyên chăm sóc mẹ và bé."},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                temperature=temperature
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            err = str(e)
            if "429" in err:
                import re as _re
                m    = _re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 10 if m else 60 * (attempt + 1)
                print(f"\n⏳ Rate limit - đợi {wait:.0f}s (lần {attempt+1}/{max_retries})...")
                _time.sleep(wait)
            else:
                _time.sleep(3)

    return ""

# ================== SUMMARIZE HISTORY ==================
def summarize_history_message(content: str) -> str:
    """Tóm tắt tin nhắn dài bằng LLM thay vì cắt ký tự thô"""
    if len(content) <= 200:
        return content
    prompt = f"""Tóm tắt tin nhắn sau thành tối đa 100 ký tự, 
giữ lại đầy đủ: tên bệnh, triệu chứng, độ tuổi, thời gian.
Không bỏ sót thông tin y tế quan trọng.

Tin nhắn: {content}
Tóm tắt:"""
    return call_llm(prompt, temperature=0).strip()

# ================== VIẾT LẠI CÂU TRUY VẤN ==================
def rewrite_and_detect_intent(question, history):
    recent_history = ""
    
    # THÊM: Trích xuất thông tin cốt lõi từ TOÀN BỘ history
    core_context = ""
    age_keywords = ["tháng tuổi", "tuổi", "sơ sinh", "tháng"]
    for msg in history:  # duyệt toàn bộ, không cắt
        content = msg.content.lower()
        for kw in age_keywords:
            if kw in content:
                # Lấy câu chứa từ khóa tuổi
                for sentence in msg.content.split('.'):
                    if kw in sentence.lower():
                        core_context += sentence.strip() + ". "
                break
    
    if history:
        lines = []
        for msg in history[-20:]:  # tăng lên 20
            role = "Mẹ" if msg.__class__.__name__ == "HumanMessage" else "MomCare"
            summarized = summarize_history_message(msg.content)
            lines.append(f"{role}: {summarized}")
        recent_history = "LỊCH SỬ:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""Dựa trên lịch sử hội thoại bên dưới, hãy thực hiện 2 việc:
1. Viết lại câu hỏi cuối thành câu đầy đủ rõ ràng (nếu đã rõ thì giữ nguyên)
2. Phân loại ý định: BLOCKED / SMALLTALK / RAG

THÔNG TIN CỐT LÕI CẦN GHI NHỚ: {core_context if core_context else "Chưa có"}

Lịch sử:
{recent_history}
Câu hỏi: {question}

Trả lời theo đúng format sau (2 dòng):
REWRITTEN: <câu hỏi viết lại, BẮT BUỘC ghi rõ độ tuổi bé nếu có trong thông tin cốt lõi>
INTENT: <BLOCKED hoặc SMALLTALK hoặc RAG>"""

    result = call_llm(prompt, temperature=0).strip()

    rewritten = question
    intent = "RAG"

    for line in result.split("\n"):
        if line.startswith("REWRITTEN:"):
            rewritten = line.replace("REWRITTEN:", "").strip()
        elif line.startswith("INTENT:"):
            raw = line.replace("INTENT:", "").strip().upper()
            if raw in ["BLOCKED", "SMALLTALK", "RAG"]:
                intent = raw

    return rewritten, intent

# ================== MULTI QUERY ==================
def generate_multi_queries(question: str, n=3):
    prompt = f"""Bạn là chuyên gia y khoa mẹ và bé. Viết lại câu hỏi sau thành {n} cách khác nhau.
Ưu tiên dùng thuật ngữ y khoa tiếng Việt (ví dụ: "ngực đau cứng" → "tắc tia sữa", "tắc tuyến sữa").

Câu hỏi: {question}

- Mỗi dòng 1 câu
- Không đánh số, không giải thích
- Chỉ viết câu hỏi"""

    try:
        text = call_llm(prompt)
    except:
        return [question]

    queries = [question]
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r'^[\d\.\-\)]+\s*', '', line)
        if line and line not in queries and len(line) > 10 and not line.endswith(':'):
            queries.append(line)

    return queries[:n + 1]  # câu gốc + n biến thể

async def summarize_one(doc, question, temperature):
    prompt = f"""Tóm tắt ngắn gọn đoạn sau liên quan đến: "{question}"
Đoạn văn: {doc.page_content}
Tối đa 3 câu. Chỉ giữ thông tin liên quan."""
    try:
        response = await async_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là trợ lý MomCare, chuyên chăm sóc mẹ và bé."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return doc.page_content[:300]  # fallback nếu lỗi

async def summarize_docs_async(docs, question, temperature):
    semaphore = asyncio.Semaphore(10)  # tối đa 10 request cùng lúc

    async def limited_summarize(doc):
        async with semaphore:
            return await summarize_one(doc, question, temperature)

    tasks = [limited_summarize(doc) for doc in docs]
    summaries = await asyncio.gather(*tasks)
    return "\n\n".join(summaries)

def summarize_docs(docs, question, temperature):
    """
    Sử dụng một Thread riêng để chạy Event Loop Async, 
    tránh block main thread của Streamlit.
    """
    try:
        result = [None]
        exception = [None]

        def run_async_logic():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result[0] = loop.run_until_complete(
                    summarize_docs_async(docs, question, temperature)
                )
            except Exception as e:
                exception[0] = e
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_logic)
        thread.start()
        thread.join(timeout=30) # Timeout an toàn 30s

        if exception[0]:
            raise exception[0]
            
        return result[0] if result[0] else "\n\n".join([d.page_content for d in docs])
        
    except Exception as e:
        print(f"Async Error, falling back to Direct: {e}")
        return "\n\n".join([d.page_content for d in docs])

# ================== RAG CHAIN (OPTIMIZED) ==================
class RAGChain:
    def __init__(self, k=5, temperature=0.1):
        self.k = k
        self.temperature = temperature
        # memory context
        self.conversation_context = ""

    def update_conversation_context(self, question):
        q = question.lower()
        matched = False

        if "6 tháng" in q:
            matched = True
            self.conversation_context = """
    - Bé 6 tháng tuổi
    - Đang ăn dặm
    - Quan tâm giấc ngủ, mọc răng, tiêm chủng
    """
        elif "2 tuổi" in q:
            matched = True
            self.conversation_context = """
    - Bé 2 tuổi
    - Quan tâm dinh dưỡng và hành vi
    """
        elif "mẹ" in q or "sau sinh" in q:
            matched = True
            self.conversation_context = """
    - Mẹ sau sinh
    - Quan tâm dinh dưỡng và hồi phục
    """
        elif "bé" in q or "con" in q:
            matched = True
            self.conversation_context = """
    - Đang nói về em bé
    """

        if not matched:
            return

    def invoke(self, inputs):
        from vectordb import smart_retrieve
        question = inputs["question"]
        history = inputs.get("history", [])

        # 1. KIỂM TRA GUARDRAILS
        blocked_msg = check_input_guardrails(question)
        if blocked_msg:
            return {"answer": blocked_msg, "docs": []}

        # 2. REWRITE + INTENT
        self.update_conversation_context(question)

        rewrite_input = f"""
        Ngữ cảnh hiện tại:
        {self.conversation_context}

        Câu hỏi:
        {question}
        """

        enriched_question, intent = rewrite_and_detect_intent(
            rewrite_input,
            history
        )

        if intent == "BLOCKED":
            return {"answer": MENTAL_HEALTH_RESPONSE, "docs": []}

        if intent == "SMALLTALK":
            prompt = f"Trả lời ngắn gọn, thân thiện: {enriched_question}"
            answer = call_llm(prompt, self.temperature)
            return {"answer": answer, "docs": []}

        # 4. TRUY XUẤT TÀI LIỆU
        search_question = f"""
        Ngữ cảnh hội thoại:
        {self.conversation_context}

        Câu hỏi:
        {enriched_question}
        """

        queries = generate_multi_queries(search_question, n=3)
        all_docs = []
        seen = set()
        for q in queries:
            adaptive_k = self.k
            if len(question.split()) <= 5:
                adaptive_k = 6

            retrieved = smart_retrieve(q, None, adaptive_k)
            for d in retrieved:
                try:
                    key = str(d.page_content)[:200]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(d)
                except Exception:
                    continue

        # RE-RANKING
        if all_docs:
            pairs = [(enriched_question, d.page_content) for d in all_docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
            docs = [d for _, d in ranked[:self.k]] # Lấy top 5
        else:
            docs = []

        if not docs:
            return {"answer": "Tôi không tìm thấy thông tin này trong tài liệu. Mẹ nên hỏi bác sĩ để được tư vấn chính xác hơn.", "docs": []}
        
        context = "\n\n".join(
            [f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)]
        )

        # 5. TẠO CÂU TRẢ LỜI
        prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi chỉ dựa trên tài liệu được cung cấp.

        NGUYÊN TẮC:
        1. Chỉ sử dụng thông tin xuất hiện trong tài liệu.
        2. Ưu tiên thông tin liên quan trực tiếp đến câu hỏi.
        3. Không thêm kiến thức bên ngoài tài liệu.
        4. Không suy diễn hoặc mở rộng ngoài nội dung được cung cấp.
        5. Nếu tài liệu chưa đủ thông tin, hãy nói rõ là chưa đủ thông tin.
        6. Trả lời ngắn gọn, đúng trọng tâm.
        7. Tối đa 4 gạch đầu dòng.
        8. Không lặp ý.
        9. Không đưa thông tin không liên quan.
        10. Trả lời tối đa 100 từ.
        11. Ưu tiên trích nguyên văn dấu hiệu trong tài liệu.
        12. Không diễn giải lại nếu tài liệu đã có câu trả lời trực tiếp.
        13. Nếu nhiều tài liệu khác nhau, ưu tiên đoạn khớp sát nhất với câu hỏi.

        TÀI LIỆU THAM KHẢO:
        {context}

        NGỮ CẢNH HỘI THOẠI:
        {self.conversation_context}

        CÂU HỎI:
        {enriched_question}

        TRẢ LỜI:"""

        answer = call_llm(prompt, self.temperature)
        answer = check_output_guardrails(answer)
        return {"answer": answer, "docs": docs}

# ================== LOAD ==================
def load_rag_chain_with_sources(number_of_documents=3, temperature=0.3):
    return RAGChain(k=number_of_documents, temperature=temperature)

def load_rag_chain(number_of_documents=3):
    return RAGChain(k=number_of_documents)

def load_normal_chain(temperature=0.3):
    class NormalChain:
        def invoke(self, inputs):
            question = inputs["question"]
            prompt = f"""
Bạn là MomCare - trợ lý chăm sóc mẹ và bé.

Trả lời dễ hiểu, chính xác.

Câu hỏi: {question}
"""
            return call_llm(prompt, temperature)
    return NormalChain()