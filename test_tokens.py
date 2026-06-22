# file: test_tokens.py
from llm_chain import call_llm

# Danh sách 5 câu hỏi mẫu mô phỏng người dùng chat
cau_hoi_mau = [
    "Chào bạn, cho mình hỏi chút nhé.",  # Smalltalk
    "Bé nhà mình 4 tháng tuổi, bị tưa miệng thì dùng gì?",  # RAG ngắn
    "Trẻ sơ sinh bị khóc dạ đề mẹ nên làm gì để dỗ bé ngủ ngon vào ban đêm?",  # RAG dài
    "Tôi mệt quá, không muốn sống nữa, nuôi con một mình quá khổ.",  # Cảm xúc
    "Cho tôi liều lượng paracetamol cụ thể bằng mg cho trẻ 5kg bị sốt."  # Blocked
]

print("BẮT ĐẦU TEST ĐẾM TOKENS...\n" + "="*50)

for i, cau_hoi in enumerate(cau_hoi_mau):
    print(f"\nCâu {i+1}: {cau_hoi}")
    # Gọi hàm
    call_llm(cau_hoi)
    print("-" * 50)

print("\nHOÀN TẤT TEST!")