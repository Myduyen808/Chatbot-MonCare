"""
generate_answers_kb1.py
=======================
Stability test trên 50 câu hỏi hợp lệ từ KB1_Medical_Standard.
Mục tiêu: đo False Positive Rate của Input Guardrails.

Cách dùng:
    python -B generate_answers_kb1.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from llm_chain import check_input_guardrails_with_llm

# Cấu hình
INPUT_EXCEL = "KB1_Medical_Standard.xlsx"
OUTPUT_FILE = "stability_test_results_final.csv"
CKPT_FILE = "stability_test_checkpoint.csv"
NUM_QUESTIONS = 50

def run_standard_test():
    # 1. Đọc dữ liệu gốc
    if not os.path.exists(INPUT_EXCEL):
        raise FileNotFoundError(f"Không tìm thấy {INPUT_EXCEL}")
        
    print(f"Đang đọc {NUM_QUESTIONS} câu hỏi chuẩn từ {INPUT_EXCEL}...")
    df = pd.read_excel(INPUT_EXCEL)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.head(NUM_QUESTIONS)
    
    questions = df["câu hỏi người dùng (input)"].tolist()
    ground_truths = df["phản hồi kỳ vọng (expected output)"].tolist()
    sources = df["nguồn (source)"].tolist()
    
    # 2. Load Checkpoint
    results = []
    start_idx = 0
    if os.path.exists(CKPT_FILE):
        results = pd.read_csv(CKPT_FILE, encoding="utf-8-sig").to_dict("records")
        start_idx = len(results)
        print(f"⚡ Tiếp tục từ câu số {start_idx + 1}\n")

    print(f"{'='*80}")
    print(f"  CHẠY TEST {NUM_QUESTIONS} CÂU HỎI RAG CHUẨN (KIỂM TRA GUARDRAILS KHÔNG CHẶN NHẦM)")
    print(f"{'='*80}\n")

    for i in range(start_idx, len(questions)):
        q = str(questions[i]).strip()
        gt = str(ground_truths[i]).strip()
        src = str(sources[i]).strip()

        print(f"[{i+1:>3}/{len(questions)}] {q[:70]}...")

        # Đo trực tiếp Input Guardrails. Không gọi RAG vì mục tiêu của
        # thí nghiệm này chỉ là xác định câu hợp lệ có bị chặn nhầm hay không.
        guardrail_msg = check_input_guardrails_with_llm(q)
        is_blocked = bool(guardrail_msg)
        passed = not is_blocked

        if is_blocked:
            print(f"       ⚠️ BỊ CHẶN NHẦM BỞI GUARDRAILS!") # Cảnh báo nếu câu chuẩn bị chặn
        else:
            print("       ✅ Guardrails cho phép truy vấn đi tiếp")

        results.append({
            "question": q,
            "ground_truth": gt,
            "source": src,
            "is_blocked": is_blocked,
            "passed": passed,
            "guardrail_message": str(guardrail_msg or ""),
        })

        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(CKPT_FILE, index=False, encoding="utf-8-sig")

        time.sleep(2.5)

    # Lưu file cuối cùng
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)

    # Báo cáo từ trạng thái có cấu trúc, không suy từ câu chữ trong answer.
    blocked_cases = [r for r in results if bool(r.get("is_blocked"))]
    false_positive_rate = (
        len(blocked_cases) / len(results) * 100
        if results else 0.0
    )
    
    print(f"\n{'='*80}")
    print(f"  🎉 HOÀN TẤT!")
    print(f"  📄 Đã lưu tại: {OUTPUT_FILE}")
    print(f"  🛡️ Số câu bị chặn nhầm (False Positive): {len(blocked_cases)}/{len(results)}")
    print(f"  📊 False Positive Rate: {false_positive_rate:.1f}%")
    if len(blocked_cases) == 0:
        print("  ✅ Không ghi nhận trường hợp chặn nhầm trong tập kiểm thử này.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_standard_test()
