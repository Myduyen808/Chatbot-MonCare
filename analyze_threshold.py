from dotenv import load_dotenv
load_dotenv()

from vectordb import load_vector_db

db = load_vector_db()

test_queries = [
    # Câu liên quan y khoa
    ("Sau khi sinh bao lâu thì em có thể bắt đầu tập các bài thể dục nhẹ nhàng để lấy lại vóc dáng ạ?", "LIÊN QUAN"),
    ("tại sao sau khi sinh xong em lại bị đau nhức khắp người, đặc biệt là vùng lưng và các khớp tay chân ạ?", "LIÊN QUAN"),
    ("Bé nhà mình hay hỏi \"Tại sao?\" liên tục, đôi khi em thấy rất mệt mỏi, có nên trả lời hết các câu hỏi đó k", "LIÊN QUAN"),
    ("Em lo ngại các hóa chất trong tã giấy, có loại tã nào k chứa gel siêu thấm hay chất tẩy trắng clo để an toàn hơn cho da bé k", "LIÊN QUAN"),
    ("Em đang định ngồi xổm chơi điện thoại trong nhà, em nghe nói sau sinh không nên ngồi xổm lâu hoặc rặn mạnh khi đi vệ sinh để tránh sa tử cung, có đúng không", "LIÊN QUAN"),
    ("Em đang tính mua máy massage lưng, em nghe nói gây tê ngoài màng cứng cũng làm mình bị bí tiểu lâu hơn phải không ạ", "LIÊN QUAN"),
    ("Em thấy sản dịch có màu đen sậm và mùi hôi kèm sốt nhẹ là dấu hiệu gì ạ?", "LIÊN QUAN"),
    ("Em bị đau núm vú quá, có cách nào để bớt đau khi cho con bú không?", "LIÊN QUAN"),
    # Câu không liên quan
    ("thời tiết hôm nay thế nào", "KHÔNG LIÊN QUAN"),
    ("giá vàng hôm nay bao nhiêu", "KHÔNG LIÊN QUAN"),
    ("công thức toán học vi tích phân", "KHÔNG LIÊN QUAN"),
    ("tin tức bóng đá hôm nay", "KHÔNG LIÊN QUAN"),
    ("phim hay nhất năm nay", "KHÔNG LIÊN QUAN"),
    ("cách nấu phở bò ngon", "KHÔNG LIÊN QUAN"),
    ("lập trình python cơ bản", "KHÔNG LIÊN QUAN"),
]

print("PHÂN TÍCH PHÂN BỐ SCORE L2 — FAISS")
print("="*65)
print(f"{'Câu hỏi':<40} {'Score':>8} {'Nhãn':<15}")
print("-"*65)

scores_lienquan = []
scores_khonglienquan = []

for q, label in test_queries:
    results = db.similarity_search_with_score(q, k=1)
    if results:
        doc, score = results[0]
        scores_lienquan.append(score) if label == "LIÊN QUAN" \
            else scores_khonglienquan.append(score)
        print(f"{q[:39]:<40} {score:>8.2f} {label}")

print("\n" + "="*65)
print("THỐNG KÊ:")
print(f"  Câu LIÊN QUAN    — Score TB: {sum(scores_lienquan)/len(scores_lienquan):.2f} "
      f"| Min: {min(scores_lienquan):.2f} | Max: {max(scores_lienquan):.2f}")
print(f"  Câu KHÔNG LIÊN   — Score TB: {sum(scores_khonglienquan)/len(scores_khonglienquan):.2f} "
      f"| Min: {min(scores_khonglienquan):.2f} | Max: {max(scores_khonglienquan):.2f}")
print(f"\n  → Ngưỡng 100 nằm giữa 2 nhóm: "
      f"Max liên quan={max(scores_lienquan):.2f} < 100 < Min không liên quan={min(scores_khonglienquan):.2f}")