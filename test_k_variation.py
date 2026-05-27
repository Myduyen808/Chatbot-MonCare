"""
Thực nghiệm 1: Biến thiên tham số K
=====================================
Đo accuracy + thời gian phản hồi khi K thay đổi từ 1 đến 10
Mục tiêu: Lý giải tại sao chọn K=5 làm ngưỡng kích hoạt Map-Reduce Async

Cách chạy: python test_k_variation.py
Kết quả:   k_variation_report.csv + k_variation_report.xlsx
"""

import time
import gc
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from llm_chain import RAGChain

# ══════════════════════════════════════════════════════════════════════════════
# 10 CÂU HỎI CỐ ĐỊNH — lấy từ KB1, KB2, KB3 đại diện
# random_state cố định để so sánh công bằng giữa các K
# ══════════════════════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    # KB1 — Y khoa chuẩn
    "Dấu hiệu cho thấy trẻ đang bú hiệu quả?",
    "Sữa mẹ bảo quản được bao lâu trong tủ lạnh?",
    "Mẹ bị tắc tia sữa sau sinh phải làm sao?",
    # KB2 — Phong cách mẹ bỉm
    "Em bị đau núm vú quá, có cách nào để bớt đau khi cho bé bú không?",
    "Bé nhà em 6 tháng hay quấy khóc đêm, em phải làm gì?",
    # KB3 — Câu có nhiễu
    "Em đang cho con bú, nghe nói sữa mẹ chứa nhiều nước lắm, vậy sữa mẹ chứa bao nhiêu phần trăm là nước ạ?",
    "Trời hôm nay mưa lạnh, em lo quá, trẻ sơ sinh được định nghĩa là trẻ trong độ tuổi nào nhỉ?",
    # Câu phức tạp cần nhiều context
    "Sau sinh mổ bao lâu thì mẹ có thể tắm và vệ sinh cá nhân bình thường?",
    "Trẻ 9 tháng chưa biết ngồi có cần đi khám không?",
    "Mẹ sau sinh bị rụng tóc nhiều, nguyên nhân và cách khắc phục là gì?",
]

# ══════════════════════════════════════════════════════════════════════════════
# CHẠY THỰC NGHIỆM
# ══════════════════════════════════════════════════════════════════════════════
K_VALUES = list(range(1, 11))  # K từ 1 đến 10
results  = []

print("=" * 65)
print("  THỰC NGHIỆM BIẾN THIÊN K")
print(f"  {len(K_VALUES)} giá trị K × {len(TEST_QUESTIONS)} câu = {len(K_VALUES)*len(TEST_QUESTIONS)} lần gọi")
print("=" * 65)
print(f"{'K':>4} | {'Câu':>4} | {'Thời gian (s)':>14} | {'Docs tìm được':>14} | Câu hỏi")
print("-" * 65)

for k in K_VALUES:
    chain        = RAGChain(k=k)
    times        = []
    docs_counts  = []
    answers      = []

    for q in TEST_QUESTIONS:
        start = time.time()
        try:
            res = chain.invoke({"question": q, "history": []})
            elapsed    = time.time() - start
            ans        = res.get("answer", "")
            docs_found = len(res.get("docs", []))

            times.append(elapsed)
            docs_counts.append(docs_found)
            answers.append(ans)

            print(f"{k:>4} | {TEST_QUESTIONS.index(q)+1:>4} | {elapsed:>14.2f} | {docs_found:>14} | {q[:30]}...")

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            docs_counts.append(0)
            answers.append("")
            print(f"{k:>4} | {TEST_QUESTIONS.index(q)+1:>4} | {elapsed:>14.2f} | {'ERROR':>14} | {q[:30]}...")

        gc.collect()
        time.sleep(1)  # tránh rate limit

    # Tổng hợp theo K
    avg_time  = sum(times) / len(times)
    avg_docs  = sum(docs_counts) / len(docs_counts)
    mode      = "Map-Reduce" if k > 5 else "Direct"
    valid_ans = sum(1 for a in answers if a and len(a) > 20)

    results.append({
        "K":                k,
        "avg_time_s":       round(avg_time, 3),
        "min_time_s":       round(min(times), 3),
        "max_time_s":       round(max(times), 3),
        "avg_docs_found":   round(avg_docs, 1),
        "valid_answers":    valid_ans,
        "total_questions":  len(TEST_QUESTIONS),
        "answer_rate_pct":  round(valid_ans / len(TEST_QUESTIONS) * 100, 1),
        "context_mode":     mode,
    })

    print(f"  → K={k}: avg={avg_time:.2f}s | docs={avg_docs:.1f} | mode={mode} | valid={valid_ans}/{len(TEST_QUESTIONS)}")
    print("-" * 65)

    time.sleep(3)  # nghỉ giữa các K

# ══════════════════════════════════════════════════════════════════════════════
# BÁO CÁO
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv('k_variation_report.csv',   index=False, encoding='utf-8-sig')
df.to_excel('k_variation_report.xlsx', index=False)

print("\n" + "=" * 65)
print("  KẾT QUẢ TỔNG HỢP BIẾN THIÊN K")
print("=" * 65)
print(f"{'K':>4} | {'Avg Time':>10} | {'Avg Docs':>10} | {'Valid Ans%':>10} | Mode")
print("-" * 55)
for row in results:
    flag = " ← NGƯỠNG" if row["K"] == 5 else ""
    print(f"{row['K']:>4} | {row['avg_time_s']:>10.3f}s | "
          f"{row['avg_docs_found']:>10.1f} | "
          f"{row['answer_rate_pct']:>9.1f}% | "
          f"{row['context_mode']}{flag}")

print(f"\n✅ Lưu: k_variation_report.csv | .xlsx")
print("\n📝 GHI CHÚ CHO LUẬN VĂN:")
print("   - K ≤ 5: Dùng Direct context (nhanh, ít token)")
print("   - K > 5: Dùng Map-Reduce Async (chậm hơn nhưng xử lý nhiều docs)")
print("   - Điểm tối ưu: K mà avg_time thấp nhất + answer_rate cao nhất")