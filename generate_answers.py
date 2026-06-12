"""
generate_answers.py
===================
Chạy chatbot MomCare trên toàn bộ KB (400 câu = 4 file x 100 câu),
lưu ra 1 file CSV duy nhất có đủ 3 cột: question | answer | ground_truth.

Cách dùng:
  # Chạy 1 KB (ghép 4 batch tự động):
  python generate_answers.py --kb kb1

  # Chỉ định file đầu ra riêng:
  python generate_answers.py --kb kb2 --output my_answers_kb2.csv

  # Chạy từ batch cụ thể (resume nếu bị ngắt giữa chừng):
  python generate_answers.py --kb kb1 --start_batch 3

Quy ước đặt tên file đầu vào (đặt cùng thư mục với script này):
  kb1_batch_1.csv, kb1_batch_2.csv, kb1_batch_3.csv, kb1_batch_4.csv
  kb2_batch_1.csv, ... (tương tự)
  kb3_batch_1.csv, ...

Mỗi file CSV đầu vào phải có 2 cột: question | ground_truth
"""

import argparse
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# PARSE ARGUMENTS
# ═══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Generate MomCare chatbot answers for evaluation")
parser.add_argument("--kb",          type=str, required=True,
                    help="Tên KB: kb1 / kb2 / kb3")
parser.add_argument("--output",      type=str, default="",
                    help="Tên file CSV đầu ra (mặc định: answers_<kb>.csv)")
parser.add_argument("--start_batch", type=int, default=1,
                    help="Bắt đầu từ batch số mấy (dùng khi resume, mặc định: 1)")
parser.add_argument("--num_batches", type=int, default=4,
                    help="Tổng số batch (mặc định: 4, tức 400 câu)")
parser.add_argument("--k",           type=int, default=5,
                    help="Số tài liệu RAG truy xuất (mặc định: 5)")
parser.add_argument("--delay",       type=float, default=2.0,
                    help="Giây nghỉ giữa các câu để tránh rate limit (mặc định: 2)")
parser.add_argument("--limit", type=int, default=0,
                    help="Giới hạn số câu chạy (0 = chạy hết, mặc định). VD: --limit 100")
args = parser.parse_args()

KB          = args.kb.lower()
OUTPUT_FILE = args.output or f"answers_{KB}.csv"
CKPT_FILE   = f"answers_checkpoint_{KB}.csv"

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
print(f"""
{'='*65}
  Generate Answers — {KB.upper()}  (batch {args.start_batch} → {args.num_batches})
  K = {args.k} docs  |  Delay = {args.delay}s/câu
  Output : {OUTPUT_FILE}
  Checkpoint: {CKPT_FILE}
{'='*65}
""")

# ═══════════════════════════════════════════════════════════════
# LOAD RAGChain
# ═══════════════════════════════════════════════════════════════
print("⏳ Đang load RAGChain + FAISS index...")
try:
    from llm_chain import RAGChain
    chain = RAGChain(k=args.k, temperature=0.1)
    print("✅ RAGChain sẵn sàng\n")
except Exception as e:
    print(f"❌ Không load được RAGChain: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# LOAD CHECKPOINT (nếu đã chạy dở)
# ═══════════════════════════════════════════════════════════════
results        = []        # list of dict
done_questions = set()     # dùng để skip câu đã xử lý

if os.path.exists(CKPT_FILE):
    ckpt_df        = pd.read_csv(CKPT_FILE, encoding="utf-8-sig")
    results        = ckpt_df.to_dict("records")
    done_questions = set(ckpt_df["question"].tolist())
    print(f"⚡ Resume: tìm thấy checkpoint với {len(results)} câu đã xử lý\n")

# ═══════════════════════════════════════════════════════════════
# HÀM GỌI CHATBOT CHO 1 CÂU HỎI
# ═══════════════════════════════════════════════════════════════
def get_answer(question: str) -> str:
    """
    Gọi RAGChain với 1 câu hỏi, trả về chuỗi answer.
    Có retry tự động nếu gặp rate limit.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "question": question,
                "history":  []      # batch eval không cần history
            })
            answer = result.get("answer", "").strip()
            if not answer:
                return "Không tìm thấy thông tin trong tài liệu."
            return answer

        except Exception as e:
            err = str(e)

            # Rate limit → chờ rồi thử lại
            if "429" in err:
                import re
                m    = re.search(r'in (\d+)m([\d.]+)s', err)
                wait = int(m.group(1)) * 60 + float(m.group(2)) + 10 if m else 60 * (attempt + 1)
                print(f"    ⏳ Rate limit — chờ {wait:.0f}s (lần {attempt+1}/{max_retries})...")
                time.sleep(wait)

            else:
                print(f"    ⚠️  Lỗi lần {attempt+1}: {err[:80]}")
                time.sleep(5)

    # Hết retry → trả về thông báo lỗi thay vì crash
    return "ERROR: Không thể lấy câu trả lời sau nhiều lần thử."

# ═══════════════════════════════════════════════════════════════
# HÀM LƯU CHECKPOINT
# ═══════════════════════════════════════════════════════════════
def save_checkpoint():
    pd.DataFrame(results).to_csv(CKPT_FILE, index=False, encoding="utf-8-sig")

# ═══════════════════════════════════════════════════════════════
# VÒNG LẶP CHÍNH — duyệt qua từng batch
# ═══════════════════════════════════════════════════════════════
total_processed = 0
total_skipped   = 0

for batch_num in range(args.start_batch, args.num_batches + 1):

    # Tìm file batch
    input_file = f"{KB}_batch_{batch_num}.csv"

    if not os.path.exists(input_file):
        print(f"⚠️  Không tìm thấy file: {input_file} — bỏ qua batch này\n")
        continue

    # Load batch
    batch_df = pd.read_csv(input_file, encoding="utf-8-sig")

    # Kiểm tra cột
    if "question" not in batch_df.columns:
        print(f"❌ File {input_file} không có cột 'question' — bỏ qua\n")
        continue

    print(f"{'─'*65}")
    print(f"  📂 BATCH {batch_num}/4  |  File: {input_file}  |  {len(batch_df)} câu")
    print(f"{'─'*65}")

    for i, row in batch_df.iterrows():
        # ── THÊM ĐOẠN NÀY ──
        if args.limit > 0 and total_processed >= args.limit:
            print(f"\n  ⏹️  Đã đạt giới hạn {args.limit} câu — dừng lại.\n")
            break
        # ── HẾT ĐOẠN THÊM ──
        question     = str(row["question"]).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()

        # Skip câu đã xử lý (resume)
        if question in done_questions:
            total_skipped += 1
            continue

        # Số thứ tự toàn cục
        global_idx = len(results) + 1
        print(f"  [{global_idx:>3}] {question[:65]}...")

        answer = get_answer(question)

        print(f"       → {answer[:90]}...")

        # Lưu kết quả
        results.append({
            "question":     question,
            "answer":       answer,
            "ground_truth": ground_truth,
            "batch":        batch_num,   # tiện theo dõi
        })
        done_questions.add(question)
        total_processed += 1

        # Lưu checkpoint mỗi 10 câu
        if total_processed % 10 == 0:
            save_checkpoint()
            print(f"\n  💾 Checkpoint: đã lưu {len(results)} câu tổng cộng\n")

        time.sleep(args.delay)

    print(f"\n  ✅ Xong batch {batch_num}: {len(batch_df)} câu\n")

    # ── THÊM ĐOẠN NÀY ──
    if args.limit > 0 and total_processed >= args.limit:
        break
    # ── HẾT ĐOẠN THÊM ──

# ═══════════════════════════════════════════════════════════════
# LƯU FILE CUỐI CÙNG
# ═══════════════════════════════════════════════════════════════
if not results:
    print("❌ Không có kết quả nào để lưu!")
    sys.exit(1)

final_df = pd.DataFrame(results)[["question", "answer", "ground_truth"]]
final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

# Xóa checkpoint sau khi lưu xong
if os.path.exists(CKPT_FILE):
    os.remove(CKPT_FILE)
    print(f"🗑️  Đã xóa checkpoint tạm")

# ═══════════════════════════════════════════════════════════════
# BÁO CÁO
# ═══════════════════════════════════════════════════════════════
error_count = sum(1 for r in results if str(r["answer"]).startswith("ERROR"))

print(f"""
{'='*65}
  ✅ HOÀN THÀNH — {KB.upper()}
{'='*65}
  Tổng câu đã xử lý : {total_processed}
  Câu skip (đã có)  : {total_skipped}
  Lỗi không trả lời : {error_count}
  File đầu ra       : {OUTPUT_FILE}
{'='*65}

Bước tiếp theo:
  python judge_clinical.py --kb {KB} --input {OUTPUT_FILE} --output final_{KB}.csv
""")