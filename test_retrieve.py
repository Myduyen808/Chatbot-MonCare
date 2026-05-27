# test_retrieve.py - chạy độc lập
from vectordb import load_vector_db

db = load_vector_db()

test_queries = [
    "Các dấu hiệu nguy hiểm ở trẻ nhỏ",
    "dấu hiệu nguy hiểm trẻ",
    "trẻ không bú được co giật khó thở"
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = db.similarity_search_with_score(q, k=3)
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc.page_content[:100]}")

# Thêm vào test_retrieve.py
out_of_scope = ["cách nấu khoai", "thời tiết hôm nay", "công thức bánh kem"]
for q in out_of_scope:
    results = db.similarity_search_with_score(q, k=1)
    print(f"Query: {q} → Score: {results[0][1]:.2f}")
    # Kỳ vọng: score > 65 → bị chặn đúng

    # Trong llm_chain.py - thay hàm is_greeting_or_smalltalk

# Câu chào thuần túy - khớp chính xác hoặc bắt đầu bằng
GREETING_EXACT = ["xin chào", "hello", "hi", "hey", "alo", "bye", "tạm biệt"]

# Câu hỏi về bot - khớp cụm từ rõ ràng  
BOT_QUESTIONS = ["bạn là ai", "bạn tên gì", "bạn làm được gì", 
                 "hỗ trợ gì", "cảm ơn", "cám ơn", "thank you"]

def is_greeting_or_smalltalk(question: str) -> bool:
    q = question.lower().strip()
    
    # Khớp chính xác câu ngắn (dưới 5 từ) với greeting
    words = q.split()
    if len(words) <= 4:
        for pattern in GREETING_EXACT:
            if pattern in q:
                return True
    
    # Khớp câu hỏi về bot - bất kể độ dài
    for pattern in BOT_QUESTIONS:
        if pattern in q:
            return True
    
    return False