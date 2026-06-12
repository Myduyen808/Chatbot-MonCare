from llm_chain import rewrite_and_detect_intent

def run_demo():
    # Danh sách các câu hỏi "ngố" để test
    test_questions = [
        "Bé bị hăm thì làm sao?",
        "Mẹ sau sinh ăn được gà không?",
        "Captopril hay Irbesartan tốt hơn?",
    ]
    
    print(f"{'CÂU HỎI GỐC':<40} | {'Ý ĐỊNH':<10} | {'CÂU HỎI ĐÃ ĐƯỢC BIẾN ĐỔI (REWRITTEN)'}")
    print("-" * 120)
    
    for q in test_questions:
        # Gọi hàm rewrite trong llm_chain của em
        rewritten, intent = rewrite_and_detect_intent(q, history=[])
        
        print(f"{q:<40} | {intent:<10} | {rewritten}")

if __name__ == "__main__":
    print("Đang khởi động hệ thống phân tích ý định...")
    run_demo()