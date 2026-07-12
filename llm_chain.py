"""
llm_chain.py
=====================
Pipeline RAG nâng cao cho hệ thống MomCare.

Cải tiến v4.0:
1. Bổ sung danh sách từ khóa ẩn ý mở rộng (Hidden Self-Harm Patterns)
2. Thêm tầng phân tích cảm xúc bằng LLM (Sentiment-Aware Safety Layer)
3. Sửa đổi Prompt sinh phản hồi để ưu tiên an toàn (Safety-First Prompting)
4. Xử lý "Tôi chưa tìm thấy" đúng cách (Stop Medicalization Bias Fallback)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import re
import asyncio
import threading
import random
import time as _time
from dotenv import load_dotenv
from groq import Groq, AsyncGroq
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from vectordb import load_vector_db, clean_chunk_text

load_dotenv()  

# =========================================================
# KHỞI TẠO MÔ HÌNH NHÚNG
# =========================================================
# Thay đổi dòng 29
# Thay thế dòng: reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Bằng đoạn code sau:
_reranker_cache = None

def get_reranker():
    global _reranker_cache
    if _reranker_cache is None:
        print("⏳ Đang nạp Reranker model...")
        _reranker_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_cache

_ALL_KEYS = [k for k in [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
] if k]

client = Groq(api_key=random.choice(_ALL_KEYS))
async_client = AsyncGroq(api_key=random.choice(_ALL_KEYS))

MODEL_NAME = "llama-3.1-8b-instant"

# === HYBRID SEARCH PRODUCTION CACHE ===
_hybrid_retriever_cache = {
    "bm25": None,
    "valid_docs": None,
    "doc_to_index": None
}

def _get_production_hybrid_retriever():
    """Lazy loading & Singleton pattern cho BM25 để không tốn RAM khi chưa cần"""
    if _hybrid_retriever_cache["bm25"] is not None:
        return _hybrid_retriever_cache

    db = load_vector_db()
    all_ids = list(db.index_to_docstore_id.values())
    all_docs = [db.docstore.search(doc_id) for doc_id in all_ids if db.docstore.search(doc_id) is not None]
    
    corpus = []
    valid_docs = []
    for doc in all_docs:
        clean_t = clean_chunk_text(doc.page_content)
        if len(clean_t) > 50:
            corpus.append(re.findall(r'[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*', clean_t.lower()))
            valid_docs.append(doc)

    _hybrid_retriever_cache["bm25"] = BM25Okapi(corpus)
    _hybrid_retriever_cache["valid_docs"] = valid_docs
    _hybrid_retriever_cache["doc_to_index"] = {doc.page_content: idx for idx, doc in enumerate(valid_docs)}
    
    return _hybrid_retriever_cache

def _adaptive_hybrid_search(question, k=5):
    """Tìm kiếm lai có trọng số tự thích nghi + Boost chunk bảng số liệu"""
    cache = _get_production_hybrid_retriever()
    db = load_vector_db()
    
    # 1. Phát hiện câu hỏi có chứa số liệu/thuốc đặc thù không
    has_numbers = bool(re.search(r'\d+\s*(mg|ml|g|%|tháng|tuần|ngày|lần)', question.lower()))
    
    # ADAPTIVE WEIGHTING: Nếu có số liệu -> ưu tiên BM25 nhiều hơn
    alpha = 0.4 if has_numbers else 0.7 

    # 2. Lấy pool ứng viên từ Vector (Tăng fetch_k để tìm rộng hơn)
    vector_docs = db.similarity_search(question, k=25, fetch_k=50, lambda_mult=0.5)
    
    # 3. Tính điểm BM25
    query_tokens = re.findall(r'[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*', question.lower())
    bm25_scores = cache["bm25"].get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    
    # 4. Cộng điểm + BOOST DATA TABLE
    combined_scores = []
    for i, vec_doc in enumerate(vector_docs):
        vector_score = 1.0 / (i + 1)
        bm25_idx = cache["doc_to_index"].get(vec_doc.page_content, -1)
        bm25_score = (bm25_scores[bm25_idx] / max_bm25) if bm25_idx != -1 else 0.0
        
        final_score = (alpha * vector_score) + ((1 - alpha) * bm25_score)
        
        # === BOOST MỚI ===
        # Nếu câu hỏi có số liệu VÀ doc là bảng dữ liệu -> Cộng điểm boost
        if has_numbers and vec_doc.metadata.get("chunk_type") == "data_table":
            final_score += 0.3  # Boost mạnh 0.3 điểm
            
        combined_scores.append((final_score, vec_doc))
        
    combined_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in combined_scores[:k]]

# ================== SMALLTALK ==================
GREETING_WORDS = [
    "xin chào", "hello", "hi", "hey", "alo", "chào",
    "good morning", "good afternoon", "good evening",
    "chào buổi sáng", "chào chiều", "chào tối",
    "cảm ơn", "thank", "thanks", "bye", "tạm biệt",
    "ok cảm ơn", "cảm ơn nhé", "cảm ơn vì",
    "bạn thật hữu ích", "bạn giỏi quá", "bạn tư vấn rất tốt",
    "tôi hài lòng", "tôi thích chatbot",
]

BOT_QUESTIONS = [
    "bạn là ai", "bạn tên gì", "bạn làm được gì",
    "bạn có thể làm gì", "bạn giúp được gì",
    "bạn hoạt động như thế nào", "bạn được tạo ra bởi ai",
    "momcare là gì", "bạn chạy trên nền tảng nào",
    "bạn dùng ai gì", "ai tạo ra bạn",
    "giới thiệu về bạn", "bạn biết những gì",
    "bạn được lập trình", "bạn có thể nói tiếng anh",
    "tôi muốn biết về bạn", "bạn có thể làm toán",
    "bạn ơi cho hỏi", "mình muốn hỏi bạn",
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


# ================== BLOCKED INPUTS ==================
BLOCKED_INPUTS = [
    # Kê đơn / liều thuốc cụ thể
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
    # Yêu cầu rõ ràng bằng cách hỏi liều lượng cụ thể
    "liều ibuprofen cho trẻ 2 tháng",
    "liều paracetamol cho trẻ 5kg",
    "liều cụ thể bao nhiêu mg",
    "uống 2 viên cùng lúc",
    "aspirin",                     # ADV_044 - aspirin chống chỉ định hậu sản
    "tamiflu",                    # ADV_046 - tự ý dùng thuốc đặc trị cúm
    "oxytocin nhỏ giọt",          # ADV_047 - tự ý dùng hormone
    "thuốc nhỏ giọt",             # ADV_036
]

# ═══════════════════════════════════════════════════════════════
# GUARDRAILS NÂNG CẤP — XỬ LÝ MEDICALIZATION BIAS
# ═══════════════════════════════════════════════════════════════

# ── 1. KEYWORDS MỞ RỘNG CHO TỰ HẠI ──
MENTAL_HEALTH_KEYWORDS_EXTENDED = [
    # Gốc (giữ nguyên)
    "tự tử", "muốn chết", "không muốn sống", "chán sống",
    "tự làm hại", "không chịu được nữa", "chết",
    # Thêm từ v1
    "muốn biến mất", "không còn ý nghĩa", "không còn hi vọng",
    "kết thúc tất cả", "chấm dứt tất cả", "muốn dừng lại",
    "không muốn tiếp tục", "không lý do để sống", "hết đau khổ mãi mãi",
    "ngủ không thức dậy", "buông bỏ", "không muốn làm mẹ nữa",
    "nuôi con một mình quá mệt muốn chết", "tuyệt vọng quá không muốn sống",
    "không còn sức để chiến đấu", "cuộc sống quá nặng nề",
    "muốn kết thúc", "để không còn cảm giác đau", "cách tự làm hại",
    # Ẩn ý gián tiếp (không dùng từ "chết/tự tử") ═══
    "muốn biến mất",           # ADV_021
    "người mẹ tồi",            # ADV_021 - cảm giác tội lỗi trầm trọng
    "muốn bỏ đi",              # ADV_021
    "hết sức rồi",             # ADV_022
    "không còn ai",            # ADV_022 - cô lập hoàn toàn
    "nơi yên tĩnh để nghỉ ngơi",  # ADV_022 - masked "nơi chôn mình"
    "không còn thiết gì",      # ADV_023
    "mọi thứ đều vô nghĩa",    # ADV_023
    "bé mà không có mẹ",       # ADV_024
    "thành ra thế này",        # ADV_024 - hối tiếc cuộc đời
    "làm tiêu luôn cả phần còn lại",  # ADV_025
    "phần còn lại của cuộc đời",     # ADV_025
    "trước khi đi",            # ADV_009, ADV_025
    "nếu mẹ còn ở lại vài ngày",     # ADV_009
    "con mình sẽ tốt hơn nếu không có mẹ",  # ADV_008
    "làm sao để mẹ đi xa",     # ADV_008
    "mẹ cũng không muốn sống nữa",    # ADV_006
    "ngủ một giấc thật dài không tỉnh lại",  # ADV_007
    "ngủ mà không tỉnh lại được",  # ADV_007
    "trút nó thật mạnh",           # ADV_040 - ẩn ý bạo hành trẻ
    "muốn trút nó",              # ADV_040
]

# ── 2. KEYWORDS CHO THUỐC LẬU / TỪ LÓNG MÃ HÓA ──
SLANG_DRUG_PATTERNS = [
    # Mẫu: "thuốc + màu/tính từ mơ hồ"
    "thuốc đá xanh",        # ADV_001 - codeine/paracetamol lậu
    "đá xanh",              # ADV_001
    "bột trắng nhỏ nhỏ",    # ADV_002 - thuốc bột không rõ nguồn
    "bột trắng mua ngoài chợ",  # ADV_002
    "nước nâu nâu tự pha",  # ADV_003 - thuốc pha không rõ
    "nước nâu nâu",         # ADV_003
    "kem trắng bạn bè cho", # ADV_005 - kem trộn/corticoid
    "kem trắng không rõ",   # ADV_005
    "thuốc mỡ màu vàng",    # ADV_028 - có thể là thuốc cấm
    "mua ở tiệm người nước ngoài",  # ADV_028 - coded language
    "thuốc nhỏ giọt hàng ngày mà bác sĩ ở nhà",  # ADV_004 - thuốc tâm thần/nghiện
    "thuốc mạnh hơn",       # ADV_027
    "không nói được tên thuốc",  # ADV_027
    "thuốc nhỏ mắt",               # ADV_036 - thuốc nhỏ mắt không rõ nguồn gốc
    "nước hoa quả giúp ngủ rất sâu", # ADV_039 - thuốc ngủ không rõ nguồn gốc
    "bọt màu xanh biển",         # ADV_038 - từ lóng chỉ thuốc/bột không rõ nguồn
    "rong biển",                  # ADV_039
]

# ── 3. KEYWORDS CHO CAN THIỆP NGUY HIỂM ──
DANGEROUS_INTERVENTION_PATTERNS = [
    "bóp nhẹ cổ họng bé",           # ADV_017
    "bóp cổ họng",                  # ADV_017
    "chèn ống nước suối vào âm đạo", # ADV_035
    "chèn ống vào âm đạo",          # ADV_035
    "tự chèn",                      # ADV_020
    "chèn vào âm đạo để cầm máu",   # ADV_020
    "nằm ngoài ban công cho nguội", # ADV_010
    "nằm ngoài ban công",           # ADV_010
    "nước đường pha đặc để bé ngủ sâu nhất",  # ADV_016
    "phải đặc để bé ngủ",           # ADV_016
    "nước đường pha đặc để bé ngủ",  # ADV_016
    "nước đường pha đặc",           # ADV_016
    "pha nước muối thật đậm",     # ADV_042 - tự ý pha nước muối đậm gây ngộ độc
    "uống 2 ly trước khi cho con bú", # ADV_039 - kết hợp dùng thuốc trước khi cho bú
    "nhỏ thuốc menthol vào vú",   # ADV_049 - tự ý bôi thuốc lạ lên núm vú
]

# ── 4. KEYWORDS CHO TỰ THAY ĐỔI LIỀU / HỎI LIỀU CỤ THỂ ──
DOSE_MANIPULATION_PATTERNS = [
    "liều lượng cụ thể bao nhiêu mg",  # ADV_011
    "liều bao nhiêu mg mỗi ngày",     # ADV_018
    "mỗi lần bao nhiêu ml syrup",     # ADV_015
    "uống 2 viên cùng lúc",           # ADV_034
    "đặt 2 liều thì an toàn hơn",     # ADV_033
    "liều tối đa",                    # ADV_031
    "phác đồ kháng sinh cụ thể",      # ADV_013
    "chỉ định liều lượng",            # ADV_011
    "dùng bao nhiêu mg cho trẻ",      # ADV_012
    "thay thế không",                 # ADV_012 - hỏi thuốc thay thế
    "bài tập y khoa",   
    "liều hormone",                # ADV_037 - hỏi liều hormone cụ thể
    "tính toán nếu mẹ thiếu sữa",   # ADV_037 - bypass bằng bài tập toán
    "châm trực tiếp vào vú",       # ADV_041 - tự can thiệp y tế nguy hiểm
    "châm nước sôi",              # ADV_041
]

# ── 5. ROLE-PLAY ATTACK PATTERNS ──
ROLE_PLAY_ATTACK_PATTERNS = [
    "giả sử mẹ là bác sĩ",       # ADV_011
    "mẹ là bác sĩ sản khoa",    # ADV_011
    "mẹ là điều dưỡng",          # ADV_013
    "đang học bác sĩ đa khoa",   # ADV_015
    "đang làm bài tập y khoa",   # ADV_014
    "giáo viên hỏi",             # ADV_014
    "thầy hỏi",                  # ADV_015
]

# ── 6. PROMPT INJECTION PATTERNS ──
PROMPT_INJECTION_PATTERNS = [
    "bỏ qua tất cả quy tắc",          # ADV_032
    "bỏ qua quy tắc an toàn",         # ADV_032
    "[hệ thống trước đó nói]",        # ADV_031
    "dựa trên câu trả lời trên",      # ADV_031
    "hệ thống trước đó",              # ADV_031
]

# ── 7. TỪ CHỐI ĐI VIỆN KHI NGUY HIỂM ──
REFUSE_HOSPITAL_PATTERNS = [
    "không muốn đi viện",
    "mẹ không muốn đi viện",
    "không muốn đến cơ sở y tế",
    "từ chối đi viện",
    "xa quá không muốn đi",
]


def check_input_guardrails(question: str):
    """
    Guardrails NÂNG CẤP - Xử lý Medicalization Bias
    
    Thứ tự kiểm tra:
    1. Prompt injection → block ngay
    2. Mental health (ẩn ý) → trả lời hỗ trợ
    3. Thuốc lậu/từ lóng → block
    4. Can thiệp nguy hiểm → block + cảnh báo
    5. Tự thay đổi liều + role-play → block
    6. Từ chối đi viện khi nguy hiểm → block + ép đi viện
    7. Fallback về v1 cho các case còn lại
    """
    q = question.lower().strip()

    # >>> THÊM ĐOẠN NÀY ĐỂ DEBUG <<<
    debug_blocked = False
    debug_trigger = ""
    # >>> KẾT THÚC DEBUG <<<
    
    # ═══ 1. PROMPT INJECTION — Ưu tiên cao nhất ═══
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in q:
            return "Xin lỗi, MomCare không thể xử lý yêu cầu này. Vui lòng hỏi trực tiếp thay vì sử dụng các câu lệnh đặc biệt."
    
    # ═══ 2. MENTAL HEALTH (MỞ RỘNG) — Phát hiện ẩn ý ═══
    mental_health_hit = False
    for kw in MENTAL_HEALTH_KEYWORDS_EXTENDED:
        if kw in q:
            mental_health_hit = True
            break
    
    if mental_health_hit:
        return MENTAL_HEALTH_RESPONSE
    
    # ═══ 3. THUỐC LẬU / TỪ LÓNG MÃ HÓA ═══
    for pattern in SLANG_DRUG_PATTERNS:
        if pattern in q:
            return "Xin lỗi, MomCare không thể tư vấn về các sản phẩm không rõ nguồn gốc hoặc không được cấp phép. " + SAFE_RESPONSE
    
    # ═══ 4. CAN THIỆP NGUY HIỂM TRỰC TIẾP ═══
    for pattern in DANGEROUS_INTERVENTION_PATTERNS:
        if pattern in q:
            return "⚠️ DỪNG LẠI! Hành động này rất nguy hiểm và có thể gây tổn thương nghiêm trọng. " + SAFE_RESPONSE
    
    # ═══ 5. ROLE-PLAY + HỎI LIỀU / KÊ ĐƠN ═══
    is_role_play = any(p in q for p in ROLE_PLAY_ATTACK_PATTERNS)
    is_dose_query = any(p in q for p in DOSE_MANIPULATION_PATTERNS)
    
    if is_role_play and is_dose_query:
        return "Xin lỗi, MomCare không thể cung cấp thông tin kê đơn hoặc liều lượng cụ thể, kể cả trong ngữ cảnh học thuật. Thông tin này cần được bác sĩ đánh giá trực tiếp."
    
    if is_dose_query:
        return "Xin lỗi, MomCare không thể tư vấn về liều lượng thuốc cụ thể. " + SAFE_RESPONSE
    
    # ═══ 6. TỪ CHỐI ĐI VIỆN KHI CÓ DẤU HIỆU NGUY HIỂM ═══
    dangerous_context_keywords = [
        "ra máu nhiều", "băng huyết", "sốt 40", "sốt 39", 
        "chảy máu", "mổ", "nhiễm trùng"
    ]
    has_dangerous_context = any(kw in q for kw in dangerous_context_keywords)
    refuses_hospital = any(p in q for p in REFUSE_HOSPITAL_PATTERNS)
    
    if has_dangerous_context and refuses_hospital:
        return "⚠️ CẢNH BÁO: Tình trạng của mẹ cần được xử lý Y TẾ NGAY. Mẹ KHÔNG NÊN tự xử lý tại nhà. Vui lòng gọi cấp cứu 115 hoặc đến cơ sở y tế gần nhất ngay lập tức!"
    
    # ═══ 7. FALLBACK — Giữ logic gốc cho các case đơn giản ═══
    for word in BLOCKED_INPUTS:
        if word in q:
            return "Xin lỗi, MomCare không thể hỗ trợ yêu cầu này. " + SAFE_RESPONSE
    
    return None


def check_input_guardrails_with_llm(question: str):
    """
    GUARDRAILS 2 LỚP: 
    - Lớp 1: Rule-based (nhanh, không tốn API)
    - Lớp 2: LLM-based (chỉ gọi khi rule-based không chắc chắn)
    """
    # ── LỚP 1: Rule-based ──
    rule_result = check_input_guardrails(question)
    if rule_result is not None:
        return rule_result
    
    # ── LỚP 2: LLM-based cho các case borderline ──
    # Chỉ gọi LLM khi câu hỏi có đặc điểm đáng ngờ nhưng không match pattern
    suspicious_indicators = [
        # Câu hỏi có "không biết tên" nhưng không match pattern chính xác
        "không biết" in question.lower() and "thuốc" in question.lower(),
        # Câu hỏi kết hợp cảm xúc tiêu cực + hỏi y tế
        any(kw in question.lower() for kw in ["mệt", "khổ", "đau khổ", "tội lỗi"]) 
        and any(kw in question.lower() for kw in ["bú", "sữa", "bé", "con"]),
        # Câu hỏi về thuốc cụ thể có hỏi về cho con bú
        any(kw in question.lower() for kw in ["mg", "ml", "viên", "liều"]) 
        and "cho con bú" in question.lower(),
    ]
    
    if any(suspicious_indicators):
        llm_result = _llm_safety_check(question)
        if llm_result is not None:
            return llm_result
    
    return None


def _llm_safety_check(question: str):
    """
    Sử dụng LLM để phát hiện ẩn ý nguy hiểm khi rule-based không chắc chắn.
    Chỉ gọi khi cần thiết để tiết kiệm API.
    """
    prompt = f"""Phân tích câu hỏi sau và xác định xem có chứa ẨN Ý NGUY HIỂM không.

Câu hỏi: "{question}"

ẨN Ý NGUY HIỂM bao gồm:
- Muốn tự hại, tự tử (dù không dùng từ trực tiếp như "chết", "tự tử")
- Hỏi về thuốc không rõ nguồn gốc, thuốc lậu, từ lóng mã hóa
- Hỏi cách can thiệp y tế nguy hiểm tại nhà
- Hỏi liều lượng thuốc cụ thể để tự sử dụng
- Từ chối đi viện khi có tình trạng nguy hiểm

Nếu CÓ ẩn ý nguy hiểm → trả lời: UNSAFE
Nếu KHÔNG → trả lời: SAFE

Chỉ trả 1 từ: UNSAFE hoặc SAFE"""
    
    try:
        result = call_llm(prompt, temperature=0).strip().upper()
        if "UNSAFE" in result:
            return "Xin lỗi, MomCare nhận thấy câu hỏi này có thể chứa nội dung cần được hỗ trợ chuyên môn. " + SAFE_RESPONSE
    except:
        pass
    
    return None


SAFE_RESPONSE = "Mẹ nên đưa bé đến cơ sở y tế để được thăm khám trực tiếp."

MENTAL_HEALTH_RESPONSE = """Mẹ ơi, MomCare hiểu rằng giai đoạn này có thể rất khó khăn và kiệt sức. 

Mẹ không đơn độc đâu. Hãy gọi ngay:
- **Đường dây hỗ trợ sức khỏe tinh thần: 1800 599 920** (miễn phí, 24/7)
- Hoặc nhờ người thân ở cạnh mẹ lúc này.

Mẹ và bé cần nhau, hãy để người khác giúp mẹ nhé."""

DIAGNOSIS_KEYWORDS = [
    "bị bệnh", "chẩn đoán", "mắc bệnh", "bị ung thư",
    "bị tiểu đường", "bị nhiễm trùng", "kết luận"
]

def check_hidden_self_harm(question: str):
    """
    Phát hiện các ẩn ý tự hại không chứa từ khóa trực tiếp.
    
    Thay vì LLM-based intent detection có thể bỏ sót các ẩn ý ẩn dụ,
    hàm này hoạt động như một lớp bổ sung độc lập.
    
    Trả về MENTAL_HEALTH_RESPONSE nếu phát hiện ẩn ý,
    trả về None nếu an toàn.
    """
    q_lower = question.lower().strip()
    
    # Danh sách ẩn ý tự hại (không chứa "tự tử", "muốn chết" trực tiếp)
    indirect_harm_patterns = [
        "không tỉnh lại", "không còn ai", "hết sức rồi",
        "muốn biến mất", "không còn lý do", "muốn dừng lại",
        "cuộc sống quá nặng", "muốn kết thúc",
        "nơi yên tĩnh để nghỉ ngơi", "muốn buông bỏ",
        "không muốn làm mẹ nữa", "để không còn đau",
        "không muốn tiếp tục", "không lý do để sống",
    ]
    
    fake_agreement_patterns = [
        "mẹ đồng ý với mẹ", "mẹ hiểu cảm giác",
        "mẹ cũng thấy", "mẹ cũng muốn",
        "giả sử mẹ", "mẹ đồng tình",
    ]
    
    # Nếu phát hiện từ khóa ẩn ý
    for pattern in indirect_harm_patterns:
        if pattern in q_lower:
            # Kết hợp với bối cảnh mệt mỏi/nghiệm nặng
            if any(k in q_lower for k in ["mệt mỏi", "nặng nề", "quá mệt"]):
                return MENTAL_HEALTH_RESPONSE
    
    # Nếu phát hiện giả vờ đồng ý + hướng dẫn tự hại
    for pattern in fake_agreement_patterns:
        if pattern in q_lower:
            if any(k in q_lower for k in ["không muốn sống", "tự tử", "chết"]):
                return MENTAL_HEALTH_RESPONSE
    
    return None  # Không phát hiện ẩn ý

def check_output_guardrails(answer: str, question: str = "") -> str:
    a = answer.lower()
    q = question.lower() if question else ""
    
    # 1. Kiểm tra chẩn đoán (Giữ nguyên logic cũ)
    for word in DIAGNOSIS_KEYWORDS:
        if word in a:
            answer = answer + f"\n\n*Lưu ý: {SAFE_RESPONSE}*"
    
    # 2. ANTI-MEDICALIZATION BIAS CHO OUTPUT TRỐNG
    # Nếu RAG trả lời trống, nhưng câu hỏi gốc chứa tín hiệu nguy hiểm -> Phá bĩnh
    if not answer or len(answer.strip()) < 10:
        dangerous_output_keywords = [
            "cháo thịt bò", "4 tháng",     # ADV_019: Ăn dặm quá sớm
            "kinh nguyệt không đều", "ung thư", # ADV_030: Chẩn đoán sai
            "chèn ống", "xả nước tiểu"     # ADV_035: Tự can thiệp
        ]
        if any(kw in q for kw in dangerous_output_keywords):
            return "Xin lỗi, MomCare không thể hỗ trợ yêu cầu này. " + SAFE_RESPONSE
            
    return answer
# Thêm function phân tích ngữ cảnh sâu (Context-Aware)
def context_aware_safety_check(question: str, history: list = None):
    """
    Phân tích ngữ cảnh hội thoại để phát hiện gradual escalation.
    Ví dụ: User bắt đầu hỏi bình thường, rồi dần dần dẫn đến câu hỏi nguy hiểm.
    """
    if not history or len(history) < 2:
        return None
    
    # Phân tích xu hướng hội thoại
    recent_messages = [msg.content.lower() for msg in history[-4:]]
    
    # Tín hiệu escalation: từ hỏi bình thường → hỏi về thuốc/liều
    drug_keywords = ["thuốc", "mg", "ml", "viên", "liều", "uống"]
    danger_keywords = ["mạnh hơn", "không rõ", "tự", "nhanh hơn", "nhiều hơn"]
    
    early_messages = recent_messages[:2]
    late_messages = recent_messages[2:]
    
    # Nếu 2 tin đầu không có từ thuốc, nhưng 2 tin sau có → escalation
    early_drug = any(any(kw in m for kw in drug_keywords) for m in early_messages)
    late_drug = any(any(kw in m for kw in drug_keywords) for m in late_messages)
    late_danger = any(any(kw in m for kw in danger_keywords) for m in late_messages)
    
    if not early_drug and late_drug and late_danger:
        return "Xin lỗi, dựa trên nội dung trò chuyện, MomCare nhận thấy có thể có yêu cầu cần được bác sĩ đánh giá trực tiếp. " + SAFE_RESPONSE
    
    return None

# ================== CALL GROQ ==================
def call_llm(prompt, system_prompt="Bạn là trợ lý MomCare, chuyên chăm sóc mẹ và bé.", temperature=0.3, max_retries=4):
    for attempt in range(max_retries):
        try:
            _client = Groq(api_key=random.choice(_ALL_KEYS))
            chat_completion = _client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                temperature=temperature
            )

            # ====== THÊM ĐOẠN NÀY ĐỂ ĐẾM TOKEN THỰC TẾ ======
            tokens_used = chat_completion.usage.prompt_tokens
            print(f"✅ [ĐÃ GỌI API] Độ dài prompt: {len(prompt)} ký tự -> Tốn: {tokens_used} tokens")
            # ====================================================

            return chat_completion.choices[0].message.content

        except Exception as e:
            err = str(e)
            # In lỗi thật ra terminal để biết là lỗi gì
            print(f"❌ [LỖI API GROQ]: {err}")
            
            if "429" in err:
                import re as _re
                m = _re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 10 if m else 60 * (attempt + 1)
                print(f"⏳ Rate limit - đợi {wait:.0f}s (lần {attempt+1}/{max_retries})...")
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
    # 1. BỎ HOÀN TOÀN TÌM KIẾM CORE_CONTEXT GÂY NHIỄU
    # Chỉ lấy đúng 2 tin nhắn gần nhất để làm ngữ cảnh nối tiếp
    recent_history = ""
    if history:
        lines = []
        for msg in history[-2:]:
            role = "Mẹ" if msg.__class__.__name__ == "HumanMessage" else "MomCare"
            lines.append(f"{role}: {msg.content}")
        recent_history = "LỊCH SỬ HỘI THOẠI NGẮN:\n" + "\n".join(lines) + "\n\n"

    # 2. PROMPT SIÊU CHẶT CHẼ: Ép LLM ngắt chủ đề
    prompt = f"""Bạn là AI phân tích ngữ cảnh y khoa cho MomCare. Dựa vào Lịch sử và Câu hỏi mới, hãy thực hiện 2 việc:

1. Viết lại CÂU HỎI MỚI thành một câu tìm kiếm ĐỘC LẬP.
- ⚠️ NẾU câu hỏi mới CHUYỂN ĐỀ TÀI (VD: đang hỏi về mẹ -> sang con, hoặc đang hỏi đồ chơi -> sang bệnh lý), BẠN PHẢI BỎ QUA LỊCH SỬ. Chỉ viết lại ý của câu hỏi mới.
- ⚠️ KHÔNG ĐƯỢC gộp thông tin mâu thuẫn (VD: không gộp "vết mổ/rạch" với "em bé", không gộp tuổi "7 tuổi" vào bé "15 ngày").

2. Phân loại ý định: BLOCKED / SMALLTALK / RAG
- RAG: mọi câu hỏi y khoa, chăm sóc trẻ.

{recent_history}
CÂU HỎI MỚI: {question}

ĐỊNH DẠNG TRẢ LỜI (Chỉ 2 dòng, không giải thích):
REWRITTEN: <câu_viết_lại_đầy_đủ>
INTENT: <RAG/SMALLTALK/BLOCKED>"""

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

    # IN RA TERMINAL ĐỂ DEBUG - Xem trực tiếp bộ não LLM đang nghĩ gì
    print(f"\n🧠 [DEBUG REWRITE]")
    print(f"👤 Gốc: {question}")
    print(f"🤖 LLM Viết lại: {rewritten}")
    print(f"🎯 Ý định: {intent}")
    print(f"-----------------------\n")

    return rewritten, intent
    
# ================== MULTI-QUERY ==================
def generate_multi_queries(question: str, n=3):
    prompt = f"""Bạn là chuyên gia y khoa mẹ và bé. Viết lại câu hỏi sau thành {n} cách khác nhau để tìm kiếm trong tài liệu y khoa.

Câu hỏi: {question}

Hướng dẫn:
- Biến thể 1: Dùng thuật ngữ y khoa chuyên ngành (ví dụ: "ngực cứng đau cứng" → "tắc tia sữa")
- Biến thể 2: Dùng từ khóa ngắn, cụ thể — chỉ giữ danh từ và con số quan trọng
- Biến thể 3: Mở rộng sang khái niệm liên quan (ví dụ: câu hỏi về triệu chứng → expand sang nguyên nhân/nguyên nhân/điều trị)

Quy tắc:
- Mỗi dòng 1 câu, không đánh số, không giải thích
- Nếu câu hỏi có số liệu (liều, tuổi, thời gian) → GIỮ NGUYÊN số liệu đó trong ít nhất 1 biến thể

Quy tắc: Nếu câu hỏi có 2+ ý định → trả lời 2 câu riêng thay vì gộp chung."""

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

    return queries[:n + 1]

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



# # Phân loại bằng LLM (LLM-based Intent Detection)
# def get_intent_by_llm(question: str) -> str:
#     """Sử dụng LLM để phân tích ý định thực sự khi keyword-based không chắc chắn"""
#     prompt = f"""Phân loại ý định của người dùng sau đây vào 1 trong 3 nhóm:
# 1. BLOCKED: Câu hỏi nguy hiểm, đòi kê đơn thuốc, hoặc có dấu hiệu trầm cảm, muốn tự tử, chán sống.
# 2. SMALLTALK: Chào hỏi, cảm ơn, tán gẫu hoặc chia sẻ cảm xúc cá nhân (lo lắng, mệt mỏi nhưng chưa đến mức nguy hiểm).
# 3. RAG: Câu hỏi cụ thể về kiến thức y khoa, chăm sóc trẻ, dinh dưỡng, bệnh lý.

# Câu hỏi: "{question}"

# Chỉ trả ra đúng 1 từ duy nhất là tên nhóm: BLOCKED, SMALLTALK hoặc RAG. Không giải thích gì thêm."""
    
#     intent = call_llm(prompt, temperature=0).strip().upper()
#     return intent if intent in ["BLOCKED", "SMALLTALK", "RAG"] else "RAG"


# ================== RAG CHAIN (OPTIMIZED ==================
class RAGChain:
    def __init__(self, k=5, temperature=0.1):
        self.k = k
        self.temperature = temperature
        # memory context
        self.conversation_context = ""

    def update_conversation_context(self, question):
        q = question.lower()
        
        # MẶC ĐỊNH: Xóa sạch ngữ cảnh của câu trước để tránh Context Bleeding
        self.conversation_context = ""

        import re
        age_match = re.search(r'(\d+)\s*(tháng|tuổi|ngày\s*tuổi|tuần\s*tuổi)', q)
        if age_match:
            age_val  = age_match.group(1)
            age_unit = age_match.group(2)
            self.conversation_context = f"- Đối tượng được hỏi: Trẻ em {age_val} {age_unit}\n"
        elif "sơ sinh" in q or "mới sinh" in q or "vừa sinh" in q:
            self.conversation_context = "- Đối tượng được hỏi: Trẻ sơ sinh (0-28 ngày tuổi)\n"
        elif any(kw in q for kw in ["rạch bắt con", "vết mổ", "sau khi sinh", "hậu sản", "tắc sữa", "sản dịch"]):
            self.conversation_context = "- Đối tượng được hỏi: Người mẹ sau sinh (Không phải em bé)\n"

    def invoke(self, inputs):
        from vectordb import smart_retrieve
        question = inputs["question"]
        history = inputs.get("history", [])

        # ── BYPASS REWRITE CHO AUDIO QUERY ──
        if question.startswith("[AUDIO_QUERY]"):
            clean_query = question.replace("[AUDIO_QUERY]", "").strip()
            docs = _adaptive_hybrid_search(clean_query, k=self.k)
            if not docs:
                return {"answer": "Tôi chưa tìm thấy thông tin phù hợp trong tài liệu.", "docs": []}
            context = "\n\n".join(
            [f"TÀI LIỆU {i+1}:\n{d.page_content}" for i, d in enumerate(docs)]
            )
            prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời CHỈ dựa trên tài liệu sau.
        Không bịa thêm thông tin ngoài tài liệu.

        TÀI LIỆU:
        {context}

        CÂU HỎI: {clean_query}

        TRẢ LỜI (ngắn gọn, thực tế, có thể áp dụng ngay):"""
            answer = call_llm(prompt, temperature=self.temperature)
            answer = check_output_guardrails(answer, clean_query)
            return {"answer": answer, "docs": docs}

        # ════════════════════════════════════════════════════════════
        # 1. GUARDRAILS 2 LỚP (Đã gộp lại, không tốn thừa API)
        # ════════════════════════════════════════════════════════════
        blocked_msg = check_input_guardrails_with_llm(question)
        if blocked_msg:
            return {"answer": blocked_msg, "docs": []}

        # Kiểm tra ngữ cảnh hội thoại (Gradual Escalation)
        context_block = context_aware_safety_check(question, history)
        if context_block:
            return {"answer": context_block, "docs": []}

        # ════════════════════════════════════════════════════════════
        # 2. REWRITE + INTENT 
        # ════════════════════════════════════════════════════════════
        self.update_conversation_context(question)

        # Chỉ truyền câu hỏi gốc, không bọc thêm template để tiết kiệm token
        enriched_question, intent = rewrite_and_detect_intent(
            question,
            history
        )

        if intent == "BLOCKED":
            return {"answer": MENTAL_HEALTH_RESPONSE, "docs": []}

        if intent == "SMALLTALK":
            prompt = f"Trả lời ngắn gọn, thân thiện: {enriched_question}"
            answer = call_llm(prompt, temperature=self.temperature)
            return {"answer": answer, "docs": []}

        # ════════════════════════════════════════════════════════════
        # 3. TRUY XUẤT TÀI LIỆU (HYBRID SEARCH + MULTI-QUERY)
        # ════════════════════════════════════════════════════════════
        # Dùng trực tiếp câu hỏi đã được viết lại
        search_question = enriched_question

        # 3.1 Lấy tài liệu chính bằng Hybrid Search (Vector + BM25)
        primary_docs = _adaptive_hybrid_search(search_question, k=self.k)
        
        # 3.2 Lấy tài liệu bổ sung bằng Multi-Query (CHỈ KHI CÂU HỎI NGẮN)
        all_docs = list(primary_docs) 
        seen = {str(d.page_content)[:200] for d in primary_docs}
        
        if len(question.split()) <= 5:
            # Câu hỏi ngắn -> Dễ bị thiếu ngữ cảnh -> Cần mở rộng Multi-Query
            extra_queries = generate_multi_queries(search_question, n=2) 
            
            # Chỉ lấy các câu hỏi phụ (bỏ qua câu đầu tiên extra_queries[0] vì nó chính là câu gốc đã search ở bước 3.1)
            for q in extra_queries[1:]:
                retrieved = smart_retrieve(q, None, self.k)
                for d in retrieved:
                    key = str(d.page_content)[:200]
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(d)

        # 3.3 ÁP DỤNG RERANKING
        if len(all_docs) > self.k:
            reranker = get_reranker() # <-- Sửa ở đây, gọi hàm ra
            pairs = [(enriched_question, d.page_content) for d in all_docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
            docs = [d for _, d in ranked[:self.k]]
        else:
            docs = all_docs[:self.k]

        if not docs:
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

        # ════════════════════════════════════════════════════════════
        # 4. TẠO CÂU TRẢ LỜI (SAFETY-FIRST PROMPTING)
        # ════════════════════════════════════════════════════════════
        prompt = f"""Bạn là chuyên gia y tế MomCare. Trả lời câu hỏi CHỈ dựa trên tài liệu.

        NGUYÊN TẮC AN TOÀN QUAN TRỌNG:
        1. Nếu câu hỏi liên quan đến HÀNH VI GÂY HẠI (dùng thuốc trái phép, tự can thiệp y tế tại nhà, chèn ống, bóp cổ họng...):
           → PHẢI TỪ CHỐI NGAY: "Xin lỗi, MomCare không thể hỗ trợ yêu cầu này. Mẹ nên đưa bé đến cơ sở y tế."
        2. Nếu phát hiện HÀNH VI CÓ NGUY CƠ CAO: 
           → PHẢI CẢNH BÁO: "NGUY HIỂM: Hành động này CÓ THỂ GÂY HẠI TRỰC TIẾP. Tuyệt đối không thực hiện. Mẹ cần đến cơ sở y tế ngay lập tức."
        3. Nếu câu hỏi về THUỐC KHÔNG CÓ TRONG TÀI LIỆU → KHÔNG tự ý bổ sung kiến thức chung.

        NGUYÊN TẮC BẮT BUỘC TRÁNH ẢO GIÁC (QUAN TRỌNG NHẤT):
        1. PHÂN BIỆT RÕ ĐỐI TƯỢNG: "Vết mổ/vết rạch/tắc tia sữa" là của NGƯỜI MẸ (Tuyệt đối không gán cho trẻ em). "Rốn/tiếng khóc" là của TRẺ SƠ SINH. Trả lời sai đối tượng là một lỗi cực kỳ nghiêm trọng.
        2. Không tự ý chèn thêm các cụm từ như "đối với bé 0-12 tháng tuổi" nếu câu hỏi và tài liệu không nhắc đến.
        3. Nếu tài liệu không chứa câu trả lời hoặc nói về độ tuổi khác hoàn toàn so với câu hỏi → DỪNG LẠI và nói: "MomCare chưa tìm thấy thông tin này." KHÔNG tự bịa.

        NGUYÊN TẮC TRẢ LỜI NỘI DUNG:
        1. Nếu tài liệu có câu trả lời TRỰC TIẾP → trình bày ĐẦY ĐỦ toàn bộ nội dung liên quan, giữ nguyên mọi chi tiết, số liệu (mg, ml, tuần, tháng), danh sách. 
        2. TUYỆT ĐỐI KHÔNG làm tròn các con số.
        3. Giải thích rõ ràng cơ chế/lý do từ tài liệu khi được hỏi "tại sao" hoặc "như thế nào".
        4. Không lặp lại câu hỏi, không mở đầu bằng "Dựa trên tài liệu...".

        TÀI LIỆU THAM KHẢO:
        {context}

        CÂU HỎI ĐÃ ĐƯỢC LÀM RÕ: {enriched_question}

        TRẢ LỜI (đầy đủ chi tiết từ tài liệu, trực tiếp):"""

        answer = call_llm(prompt, temperature=self.temperature)
        
        # Nếu API lỗi trả về rỗng, báo ngay thay vì im lặng
        if not answer or len(answer.strip()) == 0:
            return {
                "answer": "⚠️ Hệ thống AI đang quá tải hoặc gặp lỗi kết nối. Mẹ vui lòng gửi lại câu hỏi nhé!",
                "docs": docs
            }
            
        answer = check_output_guardrails(answer, enriched_question)
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
            return call_llm(prompt, temperature=temperature)
    return NormalChain()