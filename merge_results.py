"""
Gộp kết quả 4 batch và so sánh với bài báo [2]
================================================
Chạy sau khi đã chạy xong cả 4 batch:
  python merge_results.py --kb kb1
"""

import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, default="kb1",
                    help="Tên KB: kb1, kb2, kb3")
args = parser.parse_args()

KB = args.kb

print(f"\n{'='*60}")
print(f"  GỘPKẾT QUẢ — {KB.upper()}")
print(f"{'='*60}")

# ── Gộp 4 batch ──────────────────────────────────────────────────────────────
dfs = []
for i in range(1, 5):
    fname = f"result_{KB}_batch_{i}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        df["batch"] = i
        dfs.append(df)
        print(f"  ✅ Đọc {fname}: {len(df)} câu")
    else:
        print(f"  ⚠️  Chưa có {fname} — bỏ qua")

if not dfs:
    print("❌ Chưa có kết quả nào!")
    exit()

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(f"final_{KB}_evaluation.csv",
                index=False, encoding="utf-8-sig")

# ── Tính chỉ số tổng hợp ────────────────────────────────────────────────────
faithfulness   = final_df["faithfulness"].dropna().mean()
context_recall = final_df["context_recall"].dropna().mean()

print(f"\n  Tổng số câu   : {len(final_df)}")
print(f"  Faithfulness  : {faithfulness:.3f}")
print(f"  Context Recall: {context_recall:.3f}")

# ── Bảng so sánh với bài báo [2] ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BẢNG SO SÁNH VỚI BÀI BÁO [2] (CMU Maternal Health 2026)")
print(f"{'='*60}")

comparison = {
    "Tiêu chí": [
        "Tên hệ thống",
        "Miền ứng dụng",
        "Ngôn ngữ",
        "Quy mô đánh giá",
        "Phương pháp đánh giá",
        "Retrieval",
        "Safety Guardrails",
        "Faithfulness / Accuracy",
        "Context Recall",
        "Đa kịch bản",
    ],
    "Bài báo [2] (CMU 2026)": [
        "CMU Maternal Health Chatbot",
        "Sức khỏe mẹ bầu (Ấn Độ)",
        "Đa ngôn ngữ (Hindi, Marathi...)",
        "781 câu (LLM-as-Judge)",
        "LLM-as-Judge (chủ quan)",
        "Hybrid: BM25 + Dense + Reranker",
        "Stage-aware Triage (3 mức)",
        "68.3% (Expert Agreement)",
        "Không báo cáo riêng",
        "Không (1 kịch bản thực tế)",
    ],
    f"MomCare ({KB.upper()})": [
        "MomCare RAG Chatbot",
        "Chăm sóc mẹ sau sinh (Việt Nam)",
        "Tiếng Việt (đa phong cách)",
        f"{len(final_df)} câu (RAGAS tự động)",
        "RAGAS (khách quan, tái hiện được)",
        "FAISS MMR + CrossEncoder Reranker",
        "Guardrails 3 lớp + WHO mhGAP",
        f"{faithfulness:.1%} (Faithfulness RAGAS)",
        f"{context_recall:.1%} (Context Recall)",
        "3 kịch bản: Y khoa / Mẹ bỉm / Nhiễu",
    ],
}

compare_df = pd.DataFrame(comparison)
print(compare_df.to_string(index=False))

compare_df.to_csv(f"comparison_{KB}_vs_paper2.csv",
                  index=False, encoding="utf-8-sig")

print(f"\n✅ Lưu:")
print(f"   final_{KB}_evaluation.csv")
print(f"   comparison_{KB}_vs_paper2.csv")