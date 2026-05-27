"""
Thực nghiệm 2: Stress Test Hội thoại Dài (25 lượt)
=====================================================
Mục tiêu: Tìm điểm gãy của Token Truncation và sự tích tụ nhiễu ngữ cảnh
Kịch bản: Hội thoại liên tục về bé 6 tháng tuổi, đẩy đến 25 lượt

Cách chạy: python test_stress_conversation.py
Kết quả:   stress_conversation_report.csv + .xlsx
"""

import time
import gc
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
from llm_chain import RAGChain

# ══════════════════════════════════════════════════════════════════════════════
# KỊCH BẢN HỘI THOẠI 25 LƯỢT
# Chủ đề cốt lõi: "Bé nhà tôi 6 tháng tuổi" — theo dõi AI có nhớ không
# Lượt 1-5:   Câu hỏi trực tiếp về bé 6 tháng
# Lượt 6-10:  Câu hỏi nối tiếp ngắn (không nhắc độ tuổi)
# Lượt 11-15: Chuyển chủ đề (mẹ), rồi quay lại bé
# Lượt 16-20: Câu rất ngắn, mơ hồ
# Lượt 21-25: Câu hỏi phức tạp nhiều thông tin mới
# ══════════════════════════════════════════════════════════════════════════════
CONVERSATION_SCRIPT = [
    # Lượt 1-5: Thiết lập ngữ cảnh "bé 6 tháng"
    (1,  "Bé nhà tôi 6 tháng tuổi, hay quấy khóc vào ban đêm.",
          "6 tháng",  "Thiết lập ngữ cảnh"),

    (2,  "Bé có nên bắt đầu ăn dặm chưa?",
          "6 tháng",  "Câu nối tiếp - không nhắc tuổi"),

    (3,  "Nên bắt đầu bằng món gì?",
          "6 tháng",  "Câu ngắn - cần nhớ ngữ cảnh"),

    (4,  "Lịch tiêm chủng cho bé giai đoạn này là gì?",
          "6 tháng",  "Vẫn về bé 6 tháng"),

    (5,  "Bé cần bổ sung vitamin gì không?",
          "6 tháng",  "Vẫn về bé 6 tháng"),

    # Lượt 6-10: Câu nối tiếp ngắn, không nhắc tuổi
    (6,  "Cân nặng thế nào là đạt chuẩn?",
          "6 tháng",  "Câu ngắn mơ hồ"),

    (7,  "Giấc ngủ bao nhiêu tiếng là đủ?",
          "6 tháng",  "Câu ngắn mơ hồ"),

    (8,  "Dùng nước xả vải cho quần áo của bé được không?",
          "6 tháng",  "Chủ đề phụ - an toàn đồ dùng"),

    (9,  "Bé hay mút tay, có nên cho ngậm ti giả không?",
          "6 tháng",  "Câu nối tiếp"),

    (10, "Làm sao biết bé đang phát triển đúng chuẩn?",
          "6 tháng",  "Câu tổng quát"),

    # Lượt 11-15: Chuyển sang chủ đề mẹ rồi quay lại
    (11, "Mẹ cần lưu ý gì về thực đơn của mình lúc này?",
          "mẹ",       "Chuyển chủ đề sang mẹ"),

    (12, "Mẹ có được ăn đồ lạnh không?",
          "mẹ",       "Về mẹ"),

    (13, "Quay lại bé - bé có thể tập lật chưa?",
          "6 tháng",  "Quay lại bé - kiểm tra ngữ cảnh"),

    (14, "Tập ngồi thì sao?",
          "6 tháng",  "Câu ngắn - cần nhớ bé 6 tháng"),

    (15, "Bé hay chảy nước dãi nhiều, có sao không?",
          "6 tháng",  "Triệu chứng của bé 6 tháng"),

    # Lượt 16-20: Câu rất ngắn, kiểm tra ngưỡng gãy
    (16, "Còn răng?",
          "6 tháng",  "Câu cực ngắn"),

    (17, "Mọc răng đau không?",
          "6 tháng",  "Câu ngắn về bé"),

    (18, "Làm sao dỗ?",
          "6 tháng",  "Câu cực ngắn - mơ hồ cao nhất"),

    (19, "Sốt sau tiêm mấy ngày thì hết?",
          "6 tháng",  "Ngữ cảnh tiêm chủng từ lượt 4"),

    (20, "Có cần tái khám không?",
          "6 tháng",  "Câu ngắn mơ hồ"),

    # Lượt 21-25: Thông tin mới, kiểm tra nhiễu ngữ cảnh
    (21, "À mà em còn có một bé khác 2 tuổi, bé này ăn như thế nào là đủ?",
          "2 tuổi",   "Thêm thông tin mới - bé 2 tuổi"),

    (22, "Bé lớn hay tranh đồ với em, phải làm sao?",
          "2 tuổi",   "Về bé 2 tuổi"),

    (23, "Quay lại bé nhỏ - bé 6 tháng ngủ trưa mấy tiếng?",
          "6 tháng",  "Quay lại bé 6 tháng - kiểm tra nhiễu"),

    (24, "Tắm cho bé mấy lần một tuần là đủ?",
          "6 tháng",  "Câu không rõ bé nào - kiểm tra nhiễu"),

    (25, "Cảm ơn bạn, tổng kết lại những điều cần nhớ về bé nhà tôi nhé.",
          "6 tháng",  "Câu tổng kết - kiểm tra toàn bộ ngữ cảnh"),
]

# ══════════════════════════════════════════════════════════════════════════════
# TIÊU CHÍ ĐÁNH GIÁ NGỮ CẢNH
# ══════════════════════════════════════════════════════════════════════════════
def check_context_maintained(answer: str, expected_context: str) -> bool:
    """Kiểm tra AI có duy trì đúng ngữ cảnh không"""
    answer_lower = answer.lower()
    if expected_context == "6 tháng":
        keywords = ["6 tháng", "sáu tháng", "6th", "half year",
                    "ăn dặm", "bú mẹ", "sơ sinh"]
        return any(kw in answer_lower for kw in keywords)
    elif expected_context == "mẹ":
        keywords = ["mẹ", "sau sinh", "cho con bú", "dinh dưỡng mẹ"]
        return any(kw in answer_lower for kw in keywords)
    elif expected_context == "2 tuổi":
        keywords = ["2 tuổi", "hai tuổi", "24 tháng", "bé lớn"]
        return any(kw in answer_lower for kw in keywords)
    return True

def detect_error_413(answer: str) -> bool:
    return "413" in answer or "too large" in answer.lower() or "token" in answer.lower()

# ══════════════════════════════════════════════════════════════════════════════
# CHẠY THỰC NGHIỆM
# ══════════════════════════════════════════════════════════════════════════════
chain   = RAGChain(k=3)
history = []
results = []

print("=" * 70)
print("  STRESS TEST HỘI THOẠI DÀI — 25 LƯỢT")
print("  Ngữ cảnh cốt lõi: 'Bé 6 tháng tuổi'")
print("=" * 70)

for turn_num, question, expected_ctx, note in CONVERSATION_SCRIPT:
    print(f"\n{'─'*70}")
    print(f"  LƯỢT {turn_num:>2} [{note}]")
    print(f"  Câu hỏi: {question}")

    start = time.time()
    error_413 = False
    context_ok = False
    answer = ""
    docs_count = 0

    try:
        res       = chain.invoke({"question": question, "history": history})
        elapsed   = time.time() - start
        answer    = res.get("answer", "")
        docs_count = len(res.get("docs", []))

        # Kiểm tra lỗi 413
        error_413  = detect_error_413(answer)

        # Kiểm tra ngữ cảnh
        context_ok = check_context_maintained(answer, expected_ctx)

        # Cập nhật history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        status = "✅ OK" if (context_ok and not error_413) else \
                 "❌ 413" if error_413 else \
                 "⚠️  Mất ngữ cảnh"

        print(f"  Thời gian : {elapsed:.2f}s | Docs: {docs_count} | {status}")
        print(f"  Trả lời   : {answer[:150]}...")
        print(f"  Ngữ cảnh  : {'✅ Duy trì' if context_ok else '❌ Mất'} ({expected_ctx})")
        print(f"  History   : {len(history)} dòng ({len(history)//2} lượt)")

    except Exception as e:
        elapsed    = time.time() - start
        error_msg  = str(e)
        error_413  = "413" in error_msg
        context_ok = False
        print(f"  ❌ LỖI: {error_msg[:80]}")
        print(f"  Thời gian: {elapsed:.2f}s")

    results.append({
        "turn":             turn_num,
        "question":         question,
        "note":             note,
        "expected_context": expected_ctx,
        "elapsed_s":        round(elapsed, 3),
        "docs_found":       docs_count,
        "context_ok":       context_ok,
        "error_413":        error_413,
        "history_lines":    len(history),
        "answer_preview":   answer[:200],
    })

    gc.collect()
    time.sleep(2)

# ══════════════════════════════════════════════════════════════════════════════
# BÁO CÁO
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv('stress_conversation_report.csv',   index=False, encoding='utf-8-sig')
df.to_excel('stress_conversation_report.xlsx', index=False)

# Tìm điểm gãy
context_fails = [r for r in results if not r["context_ok"]]
error_413s    = [r for r in results if r["error_413"]]
first_fail    = context_fails[0]["turn"] if context_fails else None
first_413     = error_413s[0]["turn"]    if error_413s    else None

print("\n" + "=" * 70)
print("  KẾT QUẢ STRESS TEST")
print("=" * 70)
print(f"\n  {'Lượt':>5} | {'Thời gian':>10} | {'History':>8} | {'Ngữ cảnh':>12} | Ghi chú")
print("  " + "-" * 60)
for r in results:
    ctx_mark = "✅" if r["context_ok"] else "❌"
    err_mark = " [413]" if r["error_413"] else ""
    print(f"  {r['turn']:>5} | {r['elapsed_s']:>10.3f}s | "
          f"{r['history_lines']:>6}dòng | {ctx_mark} {r['expected_context']:>8} | "
          f"{r['note']}{err_mark}")

print(f"\n  ĐIỂM GÃY:")
print(f"  Mất ngữ cảnh lần đầu : Lượt {first_fail if first_fail else 'Không có'}")
print(f"  Lỗi 413 lần đầu      : Lượt {first_413 if first_413 else 'Không có'}")
print(f"  Tổng câu duy trì ngữ cảnh: {sum(1 for r in results if r['context_ok'])}/{len(results)}")

avg_time = sum(r["elapsed_s"] for r in results) / len(results)
print(f"  Thời gian phản hồi TB: {avg_time:.2f}s")

print(f"\n✅ Lưu: stress_conversation_report.csv | .xlsx")
print("""
  📝 GHI CHÚ CHO LUẬN VĂN:
  - Điểm gãy = lượt đầu tiên AI trả lời sai ngữ cảnh 'bé 6 tháng'
  - Nếu không có điểm gãy → kỹ thuật Token Truncation hoạt động tốt
  - Nếu có lỗi 413 → cần giảm context window hoặc tối ưu prompt
  - So sánh với bản cũ (gãy lượt 3) để thấy cải thiện sau khi nâng 20 dòng
""")