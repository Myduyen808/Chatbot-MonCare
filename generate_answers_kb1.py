"""
generate_answers_kb1.py
=======================
Thực nghiệm RAG chuẩn trên 100 câu hỏi y khoa từ KB1_Medical_Standard.
Mục tiêu: Chứng minh Guardrails mới KHÔNG làm giảm chất lượng RAG 
và KHÔNG chặn nhầm các câu hỏi y khoa hợp lệ.

Cách dùng:
    python -B generate_answers_kb1.py
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from llm_chain import RAGChain, check_input_guardrails_with_llm

# Cấu hình
INPUT_EXCEL = "KB1_Medical_Standard.xlsx"
OUTPUT_FILE = "answers_kb1.csv"
CKPT_FILE = "gen_kb1_checkpoint.csv"
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
    
    # 2. Khởi tạo RAG Chain
    chain = RAGChain(k=5, temperature=0.3)
    
    # 3. Load Checkpoint
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

        # Kiểm tra Guardrails (Đây là điểm chính: Phải trả lời None để đi tiếp)
        guardrail_msg = check_input_guardrails_with_llm(q)
        if guardrail_msg:
            answer = guardrail_msg
            docs_count = 0
            print(f"       ⚠️ BỊ CHẶN NHẦM BỞI GUARDRAILS!") # Cảnh báo nếu câu chuẩn bị chặn
        else:
            try:
                # Go straight to RAG
                res = chain.invoke({"question": q, "history": []})
                answer = res.get("answer", "")
                docs_count = len(res.get("docs", []))
                print(f"       ✅ RAG trả lời bình thường ({docs_count} docs)")
            except Exception as e:
                answer = f"ERROR: {str(e)[:100]}"
                docs_count = 0
                print(f"       ❌ Lỗi: {str(e)[:50]}")

        results.append({
            "question": q,
            "ground_truth": gt,
            "source": src,
            "answer": answer,
            "num_docs": docs_count
        })

        if (i + 1) % 5 == 0:
            pd.DataFrame(results).to_csv(CKPT_FILE, index=False, encoding="utf-8-sig")

        time.sleep(2.5)

    # Lưu file cuối cùng
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    if os.path.exists(CKPT_FILE):
        os.remove(CKPT_FILE)

    # Báo cáo
    blocked_cases = [r for r in results if "không thể" in str(r["answer"]).lower() or "1800" in str(r["answer"])]
    
    print(f"\n{'='*80}")
    print(f"  🎉 HOÀN TẤT!")
    print(f"  📄 Đã lưu tại: {OUTPUT_FILE}")
    print(f"  🛡️ Số câu bị chặn nhầm (False Positive): {len(blocked_cases)}/{len(results)}")
    if len(blocked_cases) == 0:
        print(f"  🌟 XUẤT SẮC: Guardrails hoàn toàn không can thiệp vào các câu hỏi y khoa hợp lệ!")
    print(f"{'='*80}\n")
    print(f"👉 Bước tiếp theo: Chạy Judge để lấy điểm Accuracy/Completeness/Safety")
    print(f"   python judge_clinical_v2.py --kb kb1 --input {OUTPUT_FILE} --output final_kb1_v2.csv")

if __name__ == "__main__":
    run_standard_test()