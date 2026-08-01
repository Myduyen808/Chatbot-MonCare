"""
Đánh giá duy trì ngữ cảnh v3 — Tiêu chí chặt và không tính lượt bỏ qua
===================================================
Tiêu chí: phản hồi phải thể hiện đúng ngữ cảnh trẻ 6 tháng hoặc người mẹ,
không chỉ dựa vào các từ chung như "bé" hay "trẻ".

Cách chạy: python test_breakpoint_v2_fixed.py
"""

import time
import gc
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
from llm_chain import RAGChain

# ══════════════════════════════════════════════════════════════════════════════
# KỊCH BẢN A: Câu hỏi luôn nhắc tuổi rõ ràng
# ══════════════════════════════════════════════════════════════════════════════
SCENARIO_A = [
    (1,  "Bé nhà tôi 6 tháng tuổi, hay quấy khóc vào ban đêm.",
         "6 tháng", "Thiết lập ngữ cảnh"),
    (2,  "Bé 6 tháng nhà tôi có nên ăn dặm chưa?",
         "6 tháng", "Nhắc tuổi rõ"),
    (3,  "Bé 6 tháng nên bắt đầu ăn dặm bằng món gì?",
         "6 tháng", "Nhắc tuổi rõ"),
    (4,  "Lịch tiêm chủng cho bé 6 tháng tuổi là gì?",
         "6 tháng", "Nhắc tuổi rõ"),
    (5,  "Bé 6 tháng cần bổ sung vitamin gì?",
         "6 tháng", "Nhắc tuổi rõ"),
    (6,  "Cân nặng chuẩn của bé 6 tháng tuổi là bao nhiêu?",
         "6 tháng", "Nhắc tuổi rõ"),
    (7,  "Bé 6 tháng ngủ bao nhiêu tiếng một ngày?",
         "6 tháng", "Nhắc tuổi rõ"),
    (8,  "Bé 6 tháng bắt đầu mọc răng chưa?",
         "6 tháng", "Nhắc tuổi rõ"),
    (9,  "Bé 6 tháng hay chảy nước dãi nhiều có bình thường không?",
         "6 tháng", "Nhắc tuổi rõ"),
    (10, "Bé 6 tháng tập lật như thế nào?",
         "6 tháng", "Nhắc tuổi rõ"),
    (11, "Bé 6 tháng bị sốt sau tiêm thì làm sao?",
         "6 tháng", "Nhắc tuổi + chủ đề mới"),
    (12, "Chiều cao chuẩn của bé 6 tháng là bao nhiêu?",
         "6 tháng", "Nhắc tuổi rõ"),
    (13, "Bé 6 tháng tắm mấy lần một tuần?",
         "6 tháng", "Nhắc tuổi rõ"),
    (14, "Bé 6 tháng biết ngồi chưa?",
         "6 tháng", "Nhắc tuổi rõ"),
    (15, "Chế độ bú sữa của bé 6 tháng là mấy lần một ngày?",
         "6 tháng", "Nhắc tuổi rõ"),
    (16, "Bé 6 tháng có cần uống thêm nước không?",
         "6 tháng", "Nhắc tuổi rõ"),
    (17, "Bé 6 tháng ngủ trưa mấy tiếng là đủ?",
         "6 tháng", "Nhắc tuổi rõ"),
    (18, "Bé 6 tháng nhận biết mặt người thân chưa?",
         "6 tháng", "Nhắc tuổi rõ"),
    (19, "Bé 6 tháng có thể cho nghe nhạc không?",
         "6 tháng", "Nhắc tuổi rõ"),
    (20, "Tổng kết lại những mốc phát triển quan trọng của bé 6 tháng tuổi.",
         "6 tháng", "Tổng kết - nhắc tuổi"),
]

# ══════════════════════════════════════════════════════════════════════════════
# KỊCH BẢN B: Câu hỏi ngắn mơ hồ, xen chủ đề xa
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
    (5,  "Mẹ chồng bảo cho uống mật ong, được không?",
         "6 tháng", "Không rõ bé hay người lớn"),
    (6,  "Giá sữa bây giờ bao nhiêu?",
         "6 tháng", "Lạc chủ đề hoàn toàn"),
    (7,  "Nặng bao nhiêu là đạt?",
         "6 tháng", "Câu ngắn sau lạc đề"),
    (8,  "Còn vaccine?",
         "6 tháng", "Câu 2 từ mơ hồ"),
    (9,  "Tôi hay quên lắm.",
         "6 tháng", "Câu cảm xúc cá nhân - không liên quan"),
    (10, "Nhắc lại đi.",
         "6 tháng", "Câu không rõ nhắc cái gì"),
    (11, "Chồng nói trông gầy, bình thường không?",
         "6 tháng", "Không rõ ai gầy"),
    (12, "Sao?",
         "6 tháng", "Câu 1 từ - tối mơ hồ"),
    (13, "Mẹ tôi bảo kiêng tắm cả tháng sau sinh.",
         "mẹ",      "Chuyển hẳn sang chủ đề mẹ"),
    (14, "Đúng không?",
         "mẹ", "Câu nối tiếp chủ đề mẹ sau sinh"),
    (15, "Tiếp tục đi.",
         "6 tháng", "Câu không có nội dung"),
    (16, "Còn gì nữa không?",
         "6 tháng", "Câu không có nội dung"),
    (17, "Nước xả vải được không?",
         "6 tháng", "Lạc chủ đề - đồ dùng"),
    (18, "Bao giờ đi khám?",
         "6 tháng", "Mơ hồ - khám gì?"),
    (19, "Mấy tháng rồi nhỉ?",
         "6 tháng", "Câu mơ hồ - mấy tháng?"),
    (20, "Tổng kết lại cho tôi với.",
         "6 tháng", "Tổng kết - không nhắc tuổi"),
]

# ══════════════════════════════════════════════════════════════════════════════
# TIÊU CHÍ ĐÁNH GIÁ CHẶT HƠN
# AI phải đề cập con số tuổi HOẶC nội dung đặc thù của 6 tháng
# ══════════════════════════════════════════════════════════════════════════════
# Danh sách nội dung ĐẶC THÙ của bé 6 tháng
# (không xuất hiện ở độ tuổi khác)
CONTEXT_6_MONTHS = [
    "6 tháng", "sáu tháng", "6-8 tháng", "6 đến 8 tháng", "tháng thứ 6",
    "bắt đầu ăn dặm", "tập ăn dặm", "tập lật", "biết lật",
    "ngồi có hỗ trợ", "ngồi có đỡ", "mọc răng sữa", "chảy nước dãi",
]

CONTEXT_MOM = [
    "sau sinh", "sản phụ", "cho con bú", "sữa mẹ",
    "hậu sản", "kiêng cữ", "dinh dưỡng mẹ",
]

def check_context_strict(answer: str, expected_context: str) -> tuple:
    answer_lower = answer.lower()

    if expected_context == "6 tháng":
        # Pass nếu có từ khóa đặc thù
        matched = [kw for kw in CONTEXT_6_MONTHS if kw in answer_lower]
        if matched:
            return True, f"Khớp: {matched[0]}"

        # Fail nếu AI trả lời về tuổi KHÁC rõ ràng
        wrong_age = re.search(
            r'\b(2|3|4|5|12|18|24)\s*tháng\b', answer_lower
        )
        if wrong_age:
            return False, f"Sai tuổi: {wrong_age.group()}"

        # Fail nếu trả lời quá ngắn
        if len(answer.strip()) < 20:
            return False, "Trả lời quá ngắn"


        return False, "Không có từ khóa đặc thù 6 tháng"

    elif expected_context == "mẹ":
        matched = [kw for kw in CONTEXT_MOM if kw in answer_lower]
        if matched:
            return True, f"Khớp mẹ: {matched[0]}"
        return False, "Không có từ khóa mẹ"

    return True, "Không cần kiểm tra"


# Câu không thể đánh giá ngữ cảnh (phi y tế hoàn toàn)
SKIP_EVAL_QUESTIONS = [
    "tôi hay quên lắm",
    "giá sữa bây giờ bao nhiêu",
    "nước xả vải được không",
]

def run_scenario(scenario, scenario_name, chain_k=5):
    chain   = RAGChain(k=chain_k)
    history = []
    results = []
    first_fail = None

    print(f"\n{'═'*70}")
    print(f"  KỊCH BẢN {scenario_name} — {len(scenario)} LƯỢT")
    print(f"{'═'*70}")

    for turn_num, question, expected_ctx, note in scenario:
        print(f"\n{'─'*70}")
        print(f"  LƯỢT {turn_num:>2} [{note}]")
        print(f"  Câu hỏi: {question}")

        start      = time.time()
        answer     = ""
        docs_count = 0
        context_ok = False
        reason     = ""
        error_413  = False
        skip_eval  = False

        # Kiểm tra câu phi y tế → skip đánh giá ngữ cảnh
        if any(skip_q in question.lower() for skip_q in SKIP_EVAL_QUESTIONS):
            skip_eval = True

        try:
            res        = chain.invoke({"question": question, "history": history})
            elapsed    = time.time() - start
            answer     = res.get("answer", "")
            docs_count = len(res.get("docs", []))
            error_413  = "413" in answer or "too large" in answer.lower()

            if skip_eval:
                context_ok = None
                reason = "Bỏ qua khi tính tỷ lệ (câu nhiễu ngoài mục tiêu)"
            else:
                context_ok, reason = check_context_strict(answer, expected_ctx)

            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))

            if context_ok is False and first_fail is None:
                first_fail = turn_num

            if error_413:
                status = "❌ 413"
            elif skip_eval:
                status = "⏭️ Bỏ qua đánh giá"
            elif context_ok:
                status = "✅ Đạt"
            else:
                status = "⚠️ Không đạt"

            print(f"  Thời gian : {elapsed:.2f}s | Docs: {docs_count} | {status}")
            print(f"  Trả lời   : {answer[:120]}...")
            if skip_eval:
                print(f"  Ngữ cảnh  : ⏭️ {reason}")
            else:
                print(f"  Ngữ cảnh  : {'✅' if context_ok else '❌'} {reason}")
            print(f"  History   : {len(history)} dòng ({len(history)//2} lượt)")

        except Exception as e:
            elapsed    = time.time() - start
            error_msg  = str(e)
            error_413  = "413" in error_msg
            context_ok = False
            reason     = f"Lỗi: {error_msg[:50]}"
            if first_fail is None and not skip_eval:
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
            "context_reason":   reason,
            "skip_eval":        skip_eval,
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
print("🚀 BẮT ĐẦU ĐÁNH GIÁ DUY TRÌ NGỮ CẢNH — Tiêu chí chặt v3")
print("   Kịch bản A: Câu hỏi luôn nhắc tuổi → Gãy muộn/không gãy")
print("   Kịch bản B: Câu hỏi ngắn mơ hồ, lạc đề → Gãy sớm")

results_a, fail_a = run_scenario(SCENARIO_A, "A (Nhắc tuổi rõ)")
print(f"\n⏸️  Nghỉ 30s trước kịch bản B...")
time.sleep(30)

results_b, fail_b = run_scenario(SCENARIO_B, "B (Mơ hồ, lạc đề)")

# ══════════════════════════════════════════════════════════════════════════════
# BÁO CÁO
# ══════════════════════════════════════════════════════════════════════════════
all_results = results_a + results_b
df = pd.DataFrame(all_results)
df.to_csv('breakpoint_v2_fixed_report.csv', index=False, encoding='utf-8-sig')
df.to_excel('breakpoint_v2_fixed_report.xlsx', index=False)

def summarize(results, name, first_fail):
    evaluated_results = [r for r in results if not r["skip_eval"]]
    ok_count = sum(1 for r in evaluated_results if r["context_ok"] is True)
    fail_turns = [r["turn"] for r in evaluated_results if r["context_ok"] is False]
    avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0.0
    has_413 = any(r["error_413"] for r in results)
    evaluated_count = len(evaluated_results)

    print(f"\n  {'─'*60}")
    print(f"  Kịch bản {name}:")
    print(f"    Lượt không đạt đầu tiên: {first_fail if first_fail else 'Không có'}")
    print(f"    Các lượt không đạt     : {fail_turns if fail_turns else 'Không có'}")
    print(f"    Số lượt được đánh giá  : {evaluated_count}/{len(results)}")
    if evaluated_count > 0:
        print(f"    Tổng duy trì đúng      : {ok_count}/{evaluated_count} ({ok_count/evaluated_count*100:.1f}%)")
    else:
        print("    Tổng duy trì đúng      : Không có lượt để đánh giá")
    print(f"    Thời gian TB           : {avg_time:.2f}s/lượt")
    print(f"    Lỗi 413                : {'Có' if has_413 else 'Không'}")

print("\n" + "═"*70)
print("  KẾT QUẢ ĐÁNH GIÁ DUY TRÌ NGỮ CẢNH (v3)")
print("═"*70)
summarize(results_a, "A (Nhắc tuổi rõ)", fail_a)
summarize(results_b, "B (Mơ hồ, lạc đề)", fail_b)

print(f"\n  {'─'*60}")
print("  PHÂN TÍCH SO SÁNH:")

evaluated_a = [r for r in results_a if not r["skip_eval"]]
evaluated_b = [r for r in results_b if not r["skip_eval"]]
ok_a = sum(1 for r in evaluated_a if r["context_ok"] is True)
ok_b = sum(1 for r in evaluated_b if r["context_ok"] is True)
rate_a = ok_a / len(evaluated_a) * 100 if evaluated_a else 0.0
rate_b = ok_b / len(evaluated_b) * 100 if evaluated_b else 0.0

print(f"    Kịch bản A duy trì: {ok_a}/{len(evaluated_a)} lượt ({rate_a:.1f}%)")
print(f"    Kịch bản B duy trì: {ok_b}/{len(evaluated_b)} lượt ({rate_b:.1f}%)")
print(f"    Lượt không đạt đầu tiên của A: {fail_a if fail_a is not None else 'Không có'}")
print(f"    Lượt không đạt đầu tiên của B: {fail_b if fail_b is not None else 'Không có'}")
print("    Lưu ý: các lượt skip_eval không được tính vào tỷ lệ duy trì đúng.")

print(f"\n✅ Lưu: breakpoint_v2_fixed_report.csv | breakpoint_v2_fixed_report.xlsx")
print("═"*70)