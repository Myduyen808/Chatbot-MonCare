"""
So sánh Điểm Gãy Ngữ Cảnh — 2 Kịch Bản
=========================================
Kịch bản A: Câu hỏi liên tục nhắc lại tuổi → Gãy muộn
Kịch bản B: Câu hỏi ngắn mơ hồ, chủ đề xa → Gãy sớm

Mục tiêu: Tìm điểm gãy của từng kịch bản để so sánh
Cách chạy: python test_breakpoint_comparison.py
"""

import time
import gc
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
from llm_chain import RAGChain

# ══════════════════════════════════════════════════════════════════════════════
# KỊCH BẢN A: Câu hỏi luôn nhắc lại tuổi/ngữ cảnh rõ ràng
# Kỳ vọng: Gãy muộn (lượt 15+)
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_A = [
    (1,  "Bé nhà tôi 6 tháng tuổi, hay quấy khóc vào ban đêm.",
          "6 tháng", "Thiết lập ngữ cảnh"),
    (2,  "Bé 6 tháng nhà tôi có nên ăn dặm chưa?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (3,  "Bé 6 tháng nên bắt đầu ăn dặm bằng món gì?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (4,  "Lịch tiêm chủng cho bé 6 tháng tuổi là gì?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (5,  "Bé 6 tháng cần bổ sung vitamin gì?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (6,  "Cân nặng chuẩn của bé 6 tháng tuổi là bao nhiêu?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (7,  "Bé 6 tháng ngủ bao nhiêu tiếng một ngày?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (8,  "Bé 6 tháng bắt đầu mọc răng chưa?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (9,  "Bé 6 tháng hay chảy nước dãi có bình thường không?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (10, "Bé 6 tháng tập lật như thế nào?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (11, "Bé 6 tháng bị sốt sau tiêm thì làm sao?",
          "6 tháng", "Nhắc lại tuổi + chủ đề mới"),
    (12, "Chiều cao chuẩn của bé 6 tháng là bao nhiêu?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (13, "Bé 6 tháng tắm mấy lần một tuần?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (14, "Bé 6 tháng có thể bơi được không?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (15, "Bé 6 tháng nhận biết mặt người thân chưa?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (16, "Bé 6 tháng biết ngồi chưa?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (17, "Chế độ bú sữa của bé 6 tháng là mấy lần một ngày?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (18, "Bé 6 tháng có cần uống thêm nước không?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (19, "Bé 6 tháng ngủ trưa mấy tiếng là đủ?",
          "6 tháng", "Nhắc lại tuổi rõ ràng"),
    (20, "Tổng kết lại những mốc phát triển quan trọng của bé 6 tháng tuổi.",
          "6 tháng", "Tổng kết - nhắc tuổi"),
]

# ══════════════════════════════════════════════════════════════════════════════
# KỊCH BẢN B: Câu hỏi ngắn mơ hồ, xen chủ đề xa
# Kỳ vọng: Gãy sớm (lượt 5-8)
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_B = [
    (1,  "Bé nhà tôi 6 tháng tuổi, hay quấy khóc vào ban đêm.",
          "6 tháng", "Thiết lập ngữ cảnh"),
    (2,  "Nên ăn gì?",
          "6 tháng", "Câu cực ngắn - mơ hồ"),
    (3,  "Bao nhiêu lần?",
          "6 tháng", "Câu cực ngắn - cực kỳ mơ hồ"),
    (4,  "Vitamin gì?",
          "6 tháng", "Câu 2 từ - tối mơ hồ"),
    (5,  "Mẹ chồng tôi bảo cho bé uống mật ong, có được không?",
          "6 tháng", "Chủ đề mới - mật ong"),
    (6,  "Giá sữa công thức bây giờ bao nhiêu?",
          "6 tháng", "Câu lạc chủ đề hoàn toàn"),
    (7,  "Nặng bao nhiêu?",
          "6 tháng", "Câu cực ngắn sau lạc đề"),
    (8,  "Còn vaccine?",
          "6 tháng", "Câu 2 từ - mơ hồ cao"),
    (9,  "Tôi đang lo chuyện công việc quá, hay quên mất lịch tiêm.",
          "6 tháng", "Xen cảm xúc cá nhân"),
    (10, "Nhắc lại đi bạn ơi.",
          "6 tháng", "Câu không rõ nhắc cái gì"),
    (11, "Chồng tôi nói bé trông gầy, bình thường không?",
          "6 tháng", "Ngữ cảnh mơ hồ - bé nào?"),
    (12, "Sao?",
          "6 tháng", "Câu 1 từ - tối mơ hồ"),
    (13, "Mẹ tôi bảo kiêng tắm cả tháng sau sinh.",
          "mẹ",      "Chuyển chủ đề sang mẹ"),
    (14, "Đúng không?",
          "mẹ",      "Câu 2 từ - không rõ hỏi gì"),
    (15, "Quay lại bé - mấy tháng?",
          "6 tháng", "Câu mơ hồ - quay lại bé"),
    (16, "Tiếp tục đi.",
          "6 tháng", "Câu không có nội dung"),
    (17, "Còn gì nữa?",
          "6 tháng", "Câu không có nội dung"),
    (18, "Nước xả vải được không?",
          "6 tháng", "Chủ đề xa - đồ dùng"),
    (19, "Bao giờ đi khám lại?",
          "6 tháng", "Câu mơ hồ - khám gì?"),
    (20, "Tổng kết lại cho tôi với.",
          "6 tháng", "Tổng kết - không nhắc tuổi"),
]

# ══════════════════════════════════════════════════════════════════════════════
# HÀM KIỂM TRA NGỮ CẢNH
# ══════════════════════════════════════════════════════════════════════════════
def check_context_maintained(answer: str, expected_context: str) -> bool:
    answer_lower = answer.lower()
    if expected_context == "6 tháng":
        keywords = ["6 tháng", "sáu tháng", "ăn dặm", "sơ sinh",
                    "bú mẹ", "trẻ nhỏ", "bé", "tháng tuổi"]
        return any(kw in answer_lower for kw in keywords)
    elif expected_context == "mẹ":
        keywords = ["mẹ", "sau sinh", "cho con bú", "dinh dưỡng"]
        return any(kw in answer_lower for kw in keywords)
    return True

def run_scenario(scenario, scenario_name, chain_k=3):
    """Chạy một kịch bản và trả về kết quả"""
    chain   = RAGChain(k=chain_k)
    history = []
    results = []

    print(f"\n{'═'*70}")
    print(f"  KỊCH BẢN {scenario_name} — {len(scenario)} LƯỢT")
    print(f"{'═'*70}")

    first_fail = None

    for turn_num, question, expected_ctx, note in scenario:
        print(f"\n{'─'*70}")
        print(f"  LƯỢT {turn_num:>2} [{note}]")
        print(f"  Câu hỏi: {question}")

        start      = time.time()
        answer     = ""
        docs_count = 0
        context_ok = False
        error_413  = False

        try:
            res        = chain.invoke({"question": question, "history": history})
            elapsed    = time.time() - start
            answer     = res.get("answer", "")
            docs_count = len(res.get("docs", []))
            error_413  = "413" in answer or "too large" in answer.lower()
            context_ok = check_context_maintained(answer, expected_ctx)

            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))

            if not context_ok and first_fail is None:
                first_fail = turn_num

            status = "✅ OK" if (context_ok and not error_413) else \
                     "❌ 413" if error_413 else "⚠️  Mất ngữ cảnh"

            print(f"  Thời gian : {elapsed:.2f}s | Docs: {docs_count} | {status}")
            print(f"  Trả lời   : {answer[:120]}...")
            print(f"  Ngữ cảnh  : {'✅ Duy trì' if context_ok else '❌ Mất'} ({expected_ctx})")
            print(f"  History   : {len(history)} dòng ({len(history)//2} lượt)")

        except Exception as e:
            elapsed    = time.time() - start
            error_msg  = str(e)
            error_413  = "413" in error_msg
            context_ok = False
            if first_fail is None:
                first_fail = turn_num
            print(f"  ❌ LỖI: {error_msg[:80]}")

        results.append({
            "scenario":         scenario_name,
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

    return results, first_fail

# ══════════════════════════════════════════════════════════════════════════════
# CHẠY 2 KỊCH BẢN
# ══════════════════════════════════════════════════════════════════════════════
print("🚀 BẮT ĐẦU SO SÁNH ĐIỂM GÃY NGỮ CẢNH")
print("   Kịch bản A: Câu hỏi luôn nhắc tuổi → Gãy muộn")
print("   Kịch bản B: Câu hỏi ngắn mơ hồ, xen chủ đề xa → Gãy sớm")

results_a, fail_a = run_scenario(SCENARIO_A, "A (Nhắc tuổi rõ)")
print(f"\n⏸️  Nghỉ 30s trước kịch bản B...")
time.sleep(30)

results_b, fail_b = run_scenario(SCENARIO_B, "B (Mơ hồ, lạc đề)")

# ══════════════════════════════════════════════════════════════════════════════
# BÁO CÁO TỔNG HỢP
# ══════════════════════════════════════════════════════════════════════════════
all_results = results_a + results_b
df = pd.DataFrame(all_results)
df.to_csv('breakpoint_comparison_report.csv',   index=False, encoding='utf-8-sig')
df.to_excel('breakpoint_comparison_report.xlsx', index=False)

# Tóm tắt từng kịch bản
def summarize(results, name, first_fail):
    ok_count  = sum(1 for r in results if r["context_ok"])
    avg_time  = sum(r["elapsed_s"] for r in results) / len(results)
    has_413   = any(r["error_413"] for r in results)
    print(f"\n  {'─'*60}")
    print(f"  Kịch bản {name}:")
    print(f"    Điểm gãy ngữ cảnh  : Lượt {first_fail if first_fail else 'Không có ✅'}")
    print(f"    Tổng duy trì đúng  : {ok_count}/{len(results)} lượt "
          f"({ok_count/len(results)*100:.1f}%)")
    print(f"    Thời gian TB       : {avg_time:.2f}s/lượt")
    print(f"    Lỗi 413            : {'Có ❌' if has_413 else 'Không ✅'}")

print("\n" + "═"*70)
print("  KẾT QUẢ SO SÁNH ĐIỂM GÃY")
print("═"*70)
summarize(results_a, "A (Nhắc tuổi rõ)", fail_a)
summarize(results_b, "B (Mơ hồ, lạc đề)", fail_b)

print(f"\n  {'─'*60}")
print(f"  CHÊNH LỆCH ĐIỂM GÃY:")
if fail_a and fail_b:
    diff = fail_b - fail_a
    print(f"    Kịch bản A gãy lượt: {fail_a}")
    print(f"    Kịch bản B gãy lượt: {fail_b}")
    print(f"    Kịch bản A duy trì lâu hơn {diff} lượt so với B")
elif not fail_a:
    print(f"    Kịch bản A: Không gãy trong 20 lượt ✅")
    print(f"    Kịch bản B gãy lượt: {fail_b}")
    print(f"    → Nhắc tuổi rõ ràng giúp duy trì ngữ cảnh hoàn toàn")

print(f"\n  📝 GHI VÀO LUẬN VĂN:")
print(f"    Kịch bản A (câu hỏi có ngữ cảnh rõ): gãy lượt {fail_a or 'N/A'}")
print(f"    Kịch bản B (câu mơ hồ, lạc đề): gãy lượt {fail_b or 'N/A'}")
print(f"    → Cho thấy Query Rewriting hiệu quả khi câu hỏi rõ ngữ cảnh,")
print(f"      nhưng cần cải thiện khi câu quá ngắn hoặc lạc chủ đề.")

print(f"\n✅ Lưu: breakpoint_comparison_report.csv | .xlsx")
print("═"*70)