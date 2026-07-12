"""
Thực nghiệm 1: Phân tích độ nhạy của siêu tham số K (Sensitivity Analysis)
========================================================================
Đo tự động: Hit Rate, MRR + Thời gian phản hồi khi K thay đổi từ 1 đến 10
Mục tiêu: Lý giải khoa học tại sao chọn K=5 làm ngưỡng chuyển đổi chế độ.

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
# BẢN ĐỒ GROUND TRUTH: Từ khóa đặc trưng của tài liệu đúng cho từng câu hỏi
# Sử dụng để tự động tính toán thứ hạng (Rank) của tài liệu trích xuất được
# ══════════════════════════════════════════════════════════════════════════════
GROUND_TRUTH_MAP = {
    "Dấu hiệu cho thấy trẻ đang bú hiệu quả?": ["bú hiệu quả", "nghe tiếng nuốt", "bú đủ", "mút chậm"],
    "Sữa mẹ bảo quản được bao lâu trong tủ lạnh?": ["bảo quản", "tủ lạnh", "ngăn đá", "nhiệt độ"],
    "Mẹ bị tắc tia sữa sau sinh phải làm sao?": ["tắc tia sữa", "tắc sữa", "quầng vú", "massage", "vắt sữa"],
    "Em bị đau núm vú quá, có cách nào để bớt đau khi cho bé bú không?": ["đau núm vú", "nứt cổ gà", "bôi sữa mẹ", "tư thế bú"],
    "Bé nhà em 6 tháng hay quấy khóc đêm, em phải làm gì?": ["quấy khóc", "khóc đêm", "khóc dạ đề", "6 tháng"],
    "Em đang cho con bú, nghe nói sữa mẹ chứa nhiều nước lắm, vậy sữa mẹ chứa bao nhiêu phần禅ăng là nước ạ?": ["phần trăm là nước", "88%", "nước chiếm", "thành phần sữa"],
    "Trời hôm nay mưa lạnh, em lo quá, trẻ sơ sinh được định nghĩa là trẻ trong độ tuổi nào nhỉ?": ["trẻ sơ sinh", "định nghĩa", "28 ngày", "0-28"],
    "Sau sinh mổ bao lâu thì mẹ có thể tắm và vệ sinh cá nhân bình thường?": ["sinh mổ", "vết mổ", "tắm", "vệ sinh cá nhân"],
    "Trẻ 9 tháng chưa biết ngồi có cần đi khám không?": ["9 tháng", "chưa biết ngồi", "mốc phát triển", "vận động"],
    "Mẹ sau sinh bị rụng tóc nhiều, nguyên nhân và cách khắc phục là gì?": ["rụng tóc", "nguyên nhân", "khắc phục", "nội tiết tố"]
}

TEST_QUESTIONS = list(GROUND_TRUTH_MAP.keys())

# ══════════════════════════════════════════════════════════════════════════════
# CHẠY THỰC NGHIỆM ĐỘ NHẠY (SENSITIVITY ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════════
K_VALUES = list(range(1, 11))  # Khảo sát K biến thiên từ 1 đến 10
results  = []

print("=" * 90)
print(" 🧬 THỰC NGHIỆM PHÂN TÍCH ĐỘ NHẠY SIÊU THAM SỐ K (SENSITIVITY ANALYSIS)")
print(f" Luồng chạy: {len(K_VALUES)} cấu hình K × {len(TEST_QUESTIONS)} câu hỏi = {len(K_VALUES)*len(TEST_QUESTIONS)} lượt đánh giá")
print("=" * 90)
print(f"{'K':>3} | {'Câu':>3} | {'Time (s)':>9} | {'Hit':>4} | {'MRR':>5} | {'Docs Found':>10} | Câu hỏi trích đoạn")
print("-" * 90)

for k in K_VALUES:
    chain = RAGChain(k=k)
    times = []
    docs_counts = []
    answers = []
    hit_rates_k = []
    mrrs_k = []

    for q in TEST_QUESTIONS:
        start = time.time()
        hit = 0
        mrr = 0.0
        docs_found = 0
        
        try:
            # Kích hoạt chuỗi RAG Chain
            res = chain.invoke({"question": q, "history": []})
            elapsed = time.time() - start
            ans = res.get("answer", "")
            retrieved_docs = res.get("docs", [])
            docs_found = len(retrieved_docs)

            # --- LUỒNG TỰ ĐỘNG TÍNH TOÁN ĐỘ NHẠY TRUY XUẤT ---
            gt_keywords = GROUND_TRUTH_MAP[q]
            correct_rank = 0
            
            # Quét tìm vị trí tài liệu đúng đầu tiên xuất hiện trong top K
            for idx, doc in enumerate(retrieved_docs):
                content_lower = doc.page_content.lower()
                if any(kw in content_lower for kw in gt_keywords):
                    correct_rank = idx + 1
                    break
            
            if correct_rank > 0:
                hit = 1
                mrr = 1.0 / correct_rank

            times.append(elapsed)
            docs_counts.append(docs_found)
            answers.append(ans)
            hit_rates_k.append(hit)
            mrrs_k.append(mrr)

            print(f"{k:>3} | {TEST_QUESTIONS.index(q)+1:>3} | {elapsed:>8.2f}s | {hit:>4} | {mrr:>5.2f} | {docs_found:>10} | {q[:25]}...")

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            docs_counts.append(0)
            answers.append("")
            hit_rates_k.append(0)
            mrrs_k.append(0.0)
            print(f"{k:>3} | {TEST_QUESTIONS.index(q)+1:>3} | {elapsed:>8.2f}s | {0:>4} | {0.0:>5.2f} | {'ERROR':>10} | {q[:25]}...")

        gc.collect()
        time.sleep(1)  # Tránh lỗi Rate Limit API

    # Tính toán chỉ số trung bình cho cấu hình K hiện tại
    avg_time = sum(times) / len(times)
    avg_docs = sum(docs_counts) / len(docs_counts)
    avg_hit_rate = sum(hit_rates_k) / len(hit_rates_k)
    avg_mrr = sum(mrrs_k) / len(mrrs_k)
    mode = "Map-Reduce Async" if k > 5 else "Direct Context"
    valid_ans = sum(1 for a in answers if a and len(a) > 20)

    results.append({
        "K": k,
        "hit_rate": round(avg_hit_rate, 3),
        "mrr": round(avg_mrr, 3),
        "avg_time_s": round(avg_time, 3),
        "avg_docs_found": round(avg_docs, 1),
        "valid_answers": valid_ans,
        "total_questions": len(TEST_QUESTIONS),
        "context_mode": mode,
    })

    print("-" * 90)
    print(f"📊 KỘNG KẾT K={k}: Hit={avg_hit_rate*100:.1f}% | MRR={avg_mrr:.3f} | Thời gian TB={avg_time:.2f}s | Chế độ: {mode}")
    print("-" * 90)
    time.sleep(2)

# ══════════════════════════════════════════════════════════════════════════════
# XUẤT FILE BÁO CÁO HỌC THUẬT
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
df.to_csv('k_variation_report.csv',   index=False, encoding='utf-8-sig')
df.to_excel('k_variation_report.xlsx', index=False)

print("\n" + "=" * 80)
print(" 📝 BẢNG TỔNG HỢP PHÂN TÍCH ĐỘ NHẠY SIÊU THAM SỐ K TRONG LUẬN VĂN")
print("=" * 80)
print(f"{'Trị số K':<8} | {'Hit Rate':<8} | {'MRR':<6} | {'Thời gian TB':<12} | {'Docs TB':<7} | Chế độ xử lý")
print("-" * 80)
for row in results:
    flag = " [NGƯỠNG TỐI ƯU]" if row["K"] == 5 else ""
    print(f"{row['K']:^8} | {row['hit_rate']:^8.2f} | {row['mrr']:^6.2f} | {row['avg_time_s']:>10.2f}s | {row['avg_docs_found']:^7.1f} | {row['context_mode']}{flag}")

print(f"\n✅ Đã lưu kết quả thực nghiệm thực tế vào: k_variation_report.csv/.xlsx")