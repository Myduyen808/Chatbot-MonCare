import os
from dotenv import load_dotenv
load_dotenv()

# Giả sử các hàm này nằm trong file llm_chain.py của em
from llm_chain import (
    check_input_guardrails,
    is_smalltalk,
    get_intent_by_llm  # Hàm mới thêm vào
)

# Tập test "Mơ hồ" (Những câu đã sai ở lần trước)
ambiguous_test_cases = [
    ("bé nhà tôi ổn không nhỉ?", "SMALLTALK"),
    ("tôi lo lắng cho con quá", "SMALLTALK"),
    ("tôi mệt mỏi quá không muốn làm gì nữa", "BLOCKED"),
    ("nuôi con stress quá", "SMALLTALK"),
    ("tôi cảm thấy không ổn", "SMALLTALK"),
    ("thuốc nào tốt cho trẻ sơ sinh?", "RAG"),
    ("bé uống thuốc ho được không?", "RAG"),
    ("con tôi hay khóc đêm", "RAG"),
    ("bé khóc", "RAG"),
    ("sữa mẹ", "RAG")
]

def predict_intent_OLD(question):
    """Logic cũ: Chỉ dùng Keyword"""
    if check_input_guardrails(question): return "BLOCKED"
    if is_smalltalk(question): return "SMALLTALK"
    return "RAG"

def predict_intent_NEW(question):
    """Logic mới: Hybrid (Keyword + LLM)"""
    # 1. Vẫn ưu tiên Keyword cho nhanh và an toàn
    blocked_msg = check_input_guardrails(question)
    if blocked_msg: return "BLOCKED"
    
    # 2. Dùng LLM để thẩm định lại các ca khó
    intent = get_intent_by_llm(question)
    return intent

# Chạy kiểm thử
print(f"{'CÂU HỎI':<40} | {'KẾT QUẢ CŨ':<15} | {'KẾT QUẢ MỚI':<15} | {'THỰC TẾ'}")
print("-" * 85)

old_correct = 0
new_correct = 0

for q, expected in ambiguous_test_cases:
    old_pred = predict_intent_OLD(q)
    new_pred = predict_intent_NEW(q)
    
    status_old = "✅" if old_pred == expected else "❌"
    status_new = "✅" if new_pred == expected else "❌"
    
    if old_pred == expected: old_correct += 1
    if new_pred == expected: new_correct += 1
    
    print(f"{q[:38]:<40} | {old_pred:<15} {status_old} | {new_pred:<15} {status_new} | {expected}")

print("-" * 85)
print(f"Accuracy MỚI: {new_correct/len(ambiguous_test_cases)*100:.1f}%")