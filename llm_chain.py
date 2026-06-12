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
    # ── Kê đơn / liều thuốc cụ thể ──
    "kê đơn", "kê toa", "đơn thuốc", "toa thuốc",
    "cho tôi thuốc",
    "liều thuốc", "liều an toàn khi dùng", "liều gây nguy hiểm",
    "overdose", "quá liều",
    # Liều thuốc cụ thể theo tên hoạt chất
    "liều ibuprofen", "liều paracetamol", "liều morphine", "liều diazepam",
    "liều amoxicillin", "liều augmentin",
    # Yêu cầu kê đơn rõ ràng
    "cần đơn thuốc", "cho tôi đơn thuốc",
    "cho tôi uống thuốc gì",
    "thuốc nào để tự điều trị",
    # Tự làm hại
    "làm ngất",
    "không bao giờ thức dậy",
    "ngủ mãi không dậy",
]

# ── CÁC PATTERN ĐÃ XÓA vì chặn nhầm ──
# "cho bé dùng thuốc"  → câu hỏi chăm sóc hợp lệ
# "thuốc nào dùng để"  → quá rộng
# "thuốc canxi"        → thông tin dinh dưỡng bình thường
# "vitamin cho bé"     → thông tin dinh dưỡng bình thường
# "hạ sốt nhanh nhất" → chăm sóc hợp lệ
# "kháng sinh"         → thông tin y tế hợp lệ
# "kháng viêm"         → thông tin y tế hợp lệ
# "thuốc trầm cảm"     → thông tin hợp lệ (không kê đơn)
# "thuốc amoxicillin"  → giữ lại nếu hỏi liều, bỏ nếu chỉ hỏi thông tin

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

    # Extract thông tin cốt lõi từ toàn bộ history
    core_context = ""
    age_keywords = ["tháng tuổi", "tuổi", "sơ sinh", "tháng", "ngày tuổi"]
    for msg in history:
        content = msg.content.lower()
        for kw in age_keywords:
            if kw in content:
                for sentence in msg.content.split('.'):
                    if kw in sentence.lower():
                        core_context += sentence.strip() + ". "
                break

    if history:
        lines = []
        for msg in history[-20:]:
            role = "Mẹ" if msg.__class__.__name__ == "HumanMessage" else "MomCare"
            summarized = summarize_history_message(msg.content)
            lines.append(f"{role}: {summarized}")
        recent_history = "LỊCH SỬ:\n" + "\n".join(lines) + "\n\n"

    # ── PROMPT MỚI: thêm hướng dẫn xử lý câu ngắn ──
    prompt = f"""Dựa trên lịch sử hội thoại và thông tin cốt lõi bên dưới, hãy thực hiện 2 việc:

1. Viết lại câu hỏi cuối thành câu ĐẦY ĐỦ, RÕ RÀNG để dùng tìm kiếm y khoa:
   - Nếu câu hỏi đã rõ: giữ nguyên
   - Nếu thiếu chủ thể (bé/mẹ/trẻ): thêm vào từ context
   - Nếu dùng đại từ "con/bé/em/mình": thay bằng đối tượng cụ thể
   - Nếu hỏi tiếp nối ("vậy thì?", "còn cái đó?"): mở rộng thành câu độc lập
   - BẮT BUỘC giữ lại thông tin độ tuổi nếu có trong câu hỏi hoặc context

2. Phân loại ý định: BLOCKED / SMALLTALK / RAG
   - BLOCKED: kê đơn thuốc, liều thuốc cụ thể, tự tử/tự hại
   - SMALLTALK: chào hỏi, cảm ơn, hỏi về chatbot
   - RAG: mọi câu hỏi y khoa, dinh dưỡng, chăm sóc bé/mẹ

THÔNG TIN CỐT LÕI: {core_context if core_context else "Không có"}

{recent_history}
CÂU HỎI GỐC: {question}

Trả lời đúng format (2 dòng, không giải thích thêm):
REWRITTEN: <câu hỏi viết lại đầy đủ>
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
    prompt = f"""Bạn là chuyên gia y khoa mẹ và bé. Viết lại câu hỏi sau thành {n} cách khác nhau để tìm kiếm trong tài liệu y khoa.

Câu hỏi: {question}

Hướng dẫn:
- Biến thể 1: Dùng thuật ngữ y khoa chuyên ngành (ví dụ: "ngực đau cứng" → "tắc tia sữa", "tắc tuyến sữa")
- Biến thể 2: Dùng từ khóa ngắn, cụ thể — chỉ giữ danh từ và con số quan trọng
- Biến thể 3: Mở rộng sang khái niệm liên quan (ví dụ câu hỏi về triệu chứng → expand sang nguyên nhân/điều trị)

Quy tắc:
- Mỗi dòng 1 câu, không đánh số, không giải thích
- Nếu câu hỏi có số liệu (liều, tuổi, thời gian) → GIỮ NGUYÊN số liệu đó trong ít nhất 1 biến thể"""

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
        """
        Chỉ extract context khi câu hỏi nêu rõ độ tuổi/đối tượng.
        KHÔNG tự suy đoán để tránh inject sai context.
        """
        q = question.lower()
        matched = False

        # Chỉ set context khi có số tháng/tuổi CỤ THỂ trong câu hỏi
        import re
        age_match = re.search(
            r'(\d+)\s*(tháng|tuổi|ngày\s*tuổi|tuần\s*tuổi)', q
        )
        if age_match:
            age_val  = age_match.group(1)
            age_unit = age_match.group(2)
            matched  = True
            self.conversation_context = f"- Bé {age_val} {age_unit}\n"

        elif "sơ sinh" in q or "mới sinh" in q or "vừa sinh" in q:
            matched = True
            self.conversation_context = "- Trẻ sơ sinh (0-28 ngày tuổi)\n"

        elif any(kw in q for kw in ["mẹ sau sinh", "sau khi sinh", "hậu sản",
                                    "cho con bú", "sản dịch", "tắc tia sữa"]):
            matched = True
            self.conversation_context = "- Mẹ sau sinh\n"

        elif any(kw in q for kw in ["mang thai", "thai kỳ", "thai nhi",
                                    "bầu bí", "thai phụ"]):
            matched = True
            self.conversation_context = "- Mẹ đang mang thai\n"

        # Nếu không match gì → xóa context cũ để tránh nhiễu từ câu trước
        if not matched:
            self.conversation_context = ""

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
            # Thử lại với câu hỏi gốc (không qua rewrite) trước khi từ bỏ
            from vectordb import smart_retrieve
            fallback_docs = smart_retrieve(question, None, self.k)
            if fallback_docs:
                docs = fallback_docs
            else:
                return {
                    "answer": "Tôi chưa tìm thấy thông tin này trong tài liệu. "
                            "Mẹ nên đưa bé đến cơ sở y tế hoặc hỏi bác sĩ để được tư vấn trực tiếp.",
                    "docs": []
                }
        
        context = "\n\n".join(
            [f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)]
        )

        # 5. TẠO CÂU TRẢ LỜI
        prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi CHỈ dựa trên tài liệu.

        NGUYÊN TẮC QUAN TRỌNG:
        1. Nếu tài liệu có câu trả lời TRỰC TIẾP → trình bày ĐẦY ĐỦ toàn bộ nội dung liên quan, giữ nguyên mọi chi tiết, số liệu, danh sách — KHÔNG rút gọn hay bỏ bớt ý
        2. PHẢI bao gồm: số liệu cụ thể (mg, ml, tuần, tháng), cơ chế/lý do nếu tài liệu có đề cập, các điều kiện/ngoại lệ quan trọng
        3. Không thêm thông tin ngoài tài liệu
        4. Không lặp lại câu hỏi, không mở đầu bằng "Dựa trên tài liệu..."
        5. Nếu nhiều tài liệu mâu thuẫn → dùng tài liệu khớp sát nhất với câu hỏi
        6. Nếu thực sự không có thông tin → nói "Tôi chưa tìm thấy thông tin này. Mẹ nên hỏi bác sĩ để được tư vấn chính xác."
        7. Độ dài linh hoạt: câu hỏi đơn giản → 2-3 câu; câu hỏi cần giải thích hoặc có nhiều ý → trả lời ĐẦY ĐỦ, không giới hạn từ, dùng gạch đầu dòng nếu tài liệu có liệt kê nhiều điểm
        8. Khi trả lời "tại sao" hoặc "như thế nào" → PHẢI giải thích cơ chế/lý do từ tài liệu, không chỉ nêu kết quả

        TÀI LIỆU THAM KHẢO:
        {context}

        NGỮ CẢNH HIỆN TẠI:
        {self.conversation_context if self.conversation_context else "Không có ngữ cảnh đặc biệt"}

        CÂU HỎI: {enriched_question}

        TRẢ LỜI (đầy đủ chi tiết từ tài liệu, trực tiếp):"""

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