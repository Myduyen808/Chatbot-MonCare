"""
╔══════════════════════════════════════════════════════════════════╗
║   MOMCARE — Đo MRR, MAP, Latency để so sánh với Bài báo số 6   ║
║   "Dual Retrieving and Ranking Medical LLM" (Sci Reports 2025)  ║
╠══════════════════════════════════════════════════════════════════╣
║  Cách dùng:                                                      ║
║  # Chạy đúng 1 batch (100 câu) — cách nhanh nhất               ║
║    python run_retrieval_metrics.py --kb kb1 --batch 1           ║
║    python run_retrieval_metrics.py --kb kb2 --batch 3           ║
║    python run_retrieval_metrics.py --kb kb3 --batch 2           ║
║                                                                  ║
║  # Chạy cả 3 KB cùng lúc, mỗi KB lấy batch 1 (100 câu)        ║
║    python run_retrieval_metrics.py --kb all --batch 1           ║
║                                                                  ║
║  # Gom nhiều batch, chỉ lấy 100 câu đầu                        ║
║    python run_retrieval_metrics.py --kb kb1 --limit 100         ║
║                                                                  ║
║  --batch 1/2/3/4 → chọn đúng file kb1_batch_X.csv              ║
║  --batch 0       → gom tất cả batch (mặc định)                 ║
║  --limit 100     → giới hạn 100 câu (mặc định)                 ║
║  --limit 0       → chạy toàn bộ không giới hạn                 ║
╚══════════════════════════════════════════════════════════════════╝

Giải thích 3 chỉ số:
  - MRR  (Mean Reciprocal Rank)  : Tài liệu đúng có đứng đầu không?
  - MAP  (Mean Average Precision): Tìm đủ và sớm các tài liệu đúng không?
  - Latency                      : Hệ thống phản hồi nhanh bao nhiêu giây?

Cấu trúc file CSV đầu vào (kb1_batch_1.csv):
  - question   : câu hỏi
  - ground_truth: câu trả lời chuẩn (dùng để tính relevance)
  - relevant_docs (tuỳ chọn): tên file tài liệu đúng, cách nhau bởi ";"
"""

import argparse
import time
import os
import warnings
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv(override=True)

# ══════════════════════════════════════════════════════════
# ARGUMENT
# ══════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--kb", type=str, default="kb1",
                    help="kb1 | kb2 | kb3 | all")
parser.add_argument("--k", type=int, default=5,
                    help="Số tài liệu retrieve (mặc định 5, đồng bộ bài báo số 6)")
parser.add_argument("--batch_size", type=int, default=100,
                    help="Số câu mỗi batch CSV (mặc định 100)")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Ngưỡng cosine similarity để tính doc là 'relevant' (mặc định 0.5)")
parser.add_argument("--limit", type=int, default=100,
                    help="Giới hạn số câu mỗi KB (mặc định 100). Dùng 0 để chạy toàn bộ.")
parser.add_argument("--batch", type=int, default=0,
                    help="Chọn batch cụ thể (1/2/3/4). Mặc định 0 = gom tất cả batch rồi lấy --limit câu.")
args = parser.parse_args()

K = args.k
SIMILARITY_THRESHOLD = args.threshold
LIMIT = args.limit if args.limit > 0 else None  # None = không giới hạn
BATCH = args.batch  # 0 = gom hết, 1-4 = chỉ chạy đúng batch đó

# ══════════════════════════════════════════════════════════
# LOAD EMBEDDING MODEL (dùng lại model của MomCare)
# ══════════════════════════════════════════════════════════
print("\n⏳ Đang load embedding model...")
from langchain_huggingface import HuggingFaceEmbeddings
import yaml

with open("model_config.yml", "r", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)

_embed_model = HuggingFaceEmbeddings(
    model_name=model_config["embedding_path"]
)
print(f"✅ Embedding model: {model_config['embedding_path']}")


# ══════════════════════════════════════════════════════════
# IMPORT MOMCARE RETRIEVER
# ══════════════════════════════════════════════════════════
from vectordb import smart_retrieve, load_vector_db

print("✅ MomCare FAISS vectordb đã sẵn sàng\n")


# ══════════════════════════════════════════════════════════
# HÀM TÍNH COSINE SIMILARITY
# ══════════════════════════════════════════════════════════
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def is_relevant_doc(doc_content: str, ground_truth: str,
                    gt_embedding: list) -> bool:
    """
    Tính relevance bằng cosine similarity giữa doc và ground_truth.
    Không cần label thủ công — tự động hoàn toàn.
    """
    doc_emb = _embed_model.embed_query(doc_content[:512])
    sim = cosine_similarity(doc_emb, gt_embedding)
    return sim >= SIMILARITY_THRESHOLD


# ══════════════════════════════════════════════════════════
# HÀM ĐO MRR, MAP, LATENCY CHO 1 CÂU HỎI
# ══════════════════════════════════════════════════════════
def evaluate_one_question(question: str, ground_truth: str, k: int):
    """
    Trả về dict: {
        'mrr_score': float,
        'ap_score' : float,   # Average Precision cho 1 câu (dùng tính MAP)
        'latency'  : float,   # giây
        'num_relevant_found': int,
        'docs_retrieved'    : list[str]  # preview nội dung doc
    }
    """
    # ── Embed ground_truth 1 lần ──
    gt_embedding = _embed_model.embed_query(ground_truth[:512])

    # ── Retrieve + đo thời gian ──
    t_start = time.perf_counter()
    docs = smart_retrieve(question, llm=None, k=k)
    t_end = time.perf_counter()
    latency = t_end - t_start

    if not docs:
        return {
            "mrr_score": 0.0,
            "ap_score": 0.0,
            "latency": latency,
            "num_relevant_found": 0,
            "docs_retrieved": []
        }

    # ── Tính relevance từng doc ──
    relevance_flags = []  # [True, False, True, ...]
    doc_previews = []

    for doc in docs:
        content = doc.page_content
        relevant = is_relevant_doc(content, ground_truth, gt_embedding)
        relevance_flags.append(relevant)
        doc_previews.append(content[:80].replace("\n", " "))

    # ── Tính MRR: 1/rank của doc đúng đầu tiên ──
    mrr_score = 0.0
    for rank, is_rel in enumerate(relevance_flags, start=1):
        if is_rel:
            mrr_score = 1.0 / rank
            break  # Chỉ tính lần đầu tiên

    # ── Tính AP (Average Precision) cho 1 câu ──
    # AP = (1/R) * sum(Precision@k * rel_k)
    # R  = tổng số doc relevant trong kết quả retrieve
    num_relevant = sum(relevance_flags)
    ap_score = 0.0
    if num_relevant > 0:
        running_relevant = 0
        precision_sum = 0.0
        for rank, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                running_relevant += 1
                precision_at_rank = running_relevant / rank
                precision_sum += precision_at_rank
        ap_score = precision_sum / num_relevant

    return {
        "mrr_score": mrr_score,
        "ap_score": ap_score,
        "latency": latency,
        "num_relevant_found": num_relevant,
        "docs_retrieved": doc_previews
    }


# ══════════════════════════════════════════════════════════
# HÀM ĐO TOÀN BỘ 1 KB
# ══════════════════════════════════════════════════════════
def run_kb(kb_name: str):
    batch_str = f"batch {BATCH}" if BATCH > 0 else "tất cả batch"
    limit_str = f" | {LIMIT} câu" if LIMIT else " | toàn bộ"
    print(f"\n{'═'*60}")
    print(f"  📊 BẮT ĐẦU ĐO: {kb_name.upper()} | {batch_str}{limit_str} | k={K}")
    print(f"{'═'*60}")

    # ── Load CSV: chọn batch cụ thể hoặc gom tất cả ──
    all_rows = []

    if BATCH > 0:
        # Chỉ load đúng batch được chỉ định
        csv_path = f"{kb_name}_batch_{BATCH}.csv"
        if not os.path.exists(csv_path):
            print(f"  ❌ Không tìm thấy file: {csv_path}")
            return None
        df_batch = pd.read_csv(csv_path, encoding="utf-8-sig")
        all_rows.append(df_batch)
        print(f"  ✅ Đã load: {csv_path} ({len(df_batch)} câu)")
    else:
        # Gom tất cả batch có sẵn
        b = 1
        while True:
            csv_path = f"{kb_name}_batch_{b}.csv"
            if not os.path.exists(csv_path):
                break
            df_batch = pd.read_csv(csv_path, encoding="utf-8-sig")
            all_rows.append(df_batch)
            print(f"  ✅ Đã load: {csv_path} ({len(df_batch)} câu)")
            b += 1

    if not all_rows:
        print(f"  ❌ Không tìm thấy file CSV nào cho {kb_name}")
        return None

    df = pd.concat(all_rows, ignore_index=True)
    print(f"\n  Tổng câu load được: {len(df)}")

    # ── Kiểm tra cột bắt buộc ──
    if "question" not in df.columns or "ground_truth" not in df.columns:
        print("  ❌ CSV phải có cột 'question' và 'ground_truth'!")
        return None

    # ── Áp dụng giới hạn số câu ──
    if LIMIT is not None:
        df = df.head(LIMIT)
        print(f"  ✂️  --limit {LIMIT}: chạy {len(df)} câu")
    else:
        print(f"  ▶️  Chạy toàn bộ {len(df)} câu")

    # ── Chạy từng câu ──
    results = []
    total = len(df)

    for i, row in df.iterrows():
        q  = str(row["question"]).strip()
        gt = str(row.get("ground_truth", "")).strip()

        if not q or not gt:
            continue

        print(f"\n  [{i+1:>3}/{total}] {q[:55]}...", end=" ", flush=True)

        try:
            res = evaluate_one_question(q, gt, K)

            mrr = res["mrr_score"]
            ap  = res["ap_score"]
            lat = res["latency"]
            n_rel = res["num_relevant_found"]

            print(f"MRR={mrr:.3f} | AP={ap:.3f} | {lat:.2f}s | rel={n_rel}/{K}")

            results.append({
                "kb"           : kb_name,
                "question"     : q,
                "ground_truth" : gt,
                "mrr_score"    : round(mrr, 4),
                "ap_score"     : round(ap, 4),
                "latency_sec"  : round(lat, 4),
                "num_relevant" : n_rel,
                "docs_preview" : " ||| ".join(res["docs_retrieved"])
            })

        except Exception as e:
            print(f"❌ Lỗi: {str(e)[:60]}")

    if not results:
        print("  ❌ Không có kết quả nào!")
        return None

    # ── Tổng hợp ──
    df_res = pd.DataFrame(results)
    mrr_mean = df_res["mrr_score"].mean()
    map_mean = df_res["ap_score"].mean()   # MAP = mean của tất cả AP
    lat_mean = df_res["latency_sec"].mean()
    lat_p95  = df_res["latency_sec"].quantile(0.95)

    summary = {
        "kb"            : kb_name,
        "total_questions": len(df_res),
        "MRR"           : round(mrr_mean, 4),
        "MAP"           : round(map_mean, 4),
        "Latency_mean_s": round(lat_mean, 4),
        "Latency_p95_s" : round(lat_p95, 4),
    }

    # ── Lưu chi tiết từng câu ──
    detail_path = f"retrieval_detail_{kb_name}.csv"
    df_res.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Chi tiết đã lưu: {detail_path}")

    # ── In kết quả ──
    print(f"\n  {'─'*50}")
    print(f"  KẾT QUẢ {kb_name.upper()} (k={K})")
    print(f"  {'─'*50}")
    print(f"  Số câu đánh giá : {summary['total_questions']}")
    print(f"  MRR             : {summary['MRR']:.4f}")
    print(f"  MAP             : {summary['MAP']:.4f}")
    print(f"  Latency (mean)  : {summary['Latency_mean_s']:.4f}s")
    print(f"  Latency (P95)   : {summary['Latency_p95_s']:.4f}s")
    print(f"  {'─'*50}")

    return summary


# ══════════════════════════════════════════════════════════
# BẢNG SO SÁNH VỚI BÀI BÁO SỐ 6
# ══════════════════════════════════════════════════════════
# Dữ liệu từ Table 4 của bài báo số 6 (Sci Reports 2025)
# Yang et al., "Dual retrieving and ranking medical LLM"
PAPER6_BENCHMARK = {
    "System":     "Bài báo số 6 (Dual Retrieval + ColBERTv2)",
    "MRR":        0.72,    # Số thực từ Table 4
    "MAP":        0.63,    # Số thực từ Table 4
    "Latency_s":  None,
    "Note":       "Kiến trúc: Chroma + Elasticsearch + ColBERTv2 | GPU: NVIDIA A40"
}

def print_comparison_table(summaries: list):
    """In bảng so sánh MomCare vs Bài báo số 6"""

    print(f"\n{'═'*70}")
    print(f"  📋 BẢNG SO SÁNH MOMCARE vs BÀI BÁO SỐ 6")
    print(f"  (Yang et al., Scientific Reports 2025)")
    print(f"{'═'*70}")

    # Header
    print(f"  {'Hệ thống':<35} {'MRR':>8} {'MAP':>8} {'Latency':>12}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*12}")

    # Bài báo số 6
    lat_paper = str(PAPER6_BENCHMARK['Latency_s']) + "s" if PAPER6_BENCHMARK['Latency_s'] else "N/A"
    print(f"  {'[REF6] Dual Retrieval+ColBERTv2':<35} "
          f"{PAPER6_BENCHMARK['MRR']:>8.4f} "
          f"{PAPER6_BENCHMARK['MAP']:>8.4f} "
          f"{lat_paper:>12}")

    # MomCare từng KB
    for s in summaries:
        kb  = s["kb"].upper()
        mrr = s["MRR"]
        map_ = s["MAP"]
        lat = f"{s['Latency_mean_s']:.4f}s"
        delta_mrr = mrr - PAPER6_BENCHMARK["MRR"]
        delta_map = map_ - PAPER6_BENCHMARK["MAP"]
        arrow_mrr = "▲" if delta_mrr >= 0 else "▼"
        arrow_map = "▲" if delta_map >= 0 else "▼"
        label = f"[MomCare] FAISS+MMR ({kb})"
        print(f"  {label:<35} "
              f"{mrr:>8.4f} "
              f"{map_:>8.4f} "
              f"{lat:>12}  "
              f"{arrow_mrr}MRR={delta_mrr:+.4f} {arrow_map}MAP={delta_map:+.4f}")

    print(f"  {'─'*70}")
    print(f"\n  📝 Ghi chú:")
    print(f"  - Bài báo số 6: {PAPER6_BENCHMARK['Note']}")
    print(f"  - MomCare: FAISS + MMR (lambda_mult=0.7) + CrossEncoder reranker")
    print(f"  - KB1=câu y thuần túy | KB2=ngôn ngữ mẹ bỉm | KB3=có câu nhiễu")
    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    kb_arg = args.kb.lower()

    if kb_arg == "all":
        kb_list = ["kb1", "kb2", "kb3"]
    else:
        kb_list = [kb_arg]

    summaries = []
    for kb in kb_list:
        result = run_kb(kb)
        if result:
            summaries.append(result)

    if summaries:
        # Lưu summary tổng hợp
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv("retrieval_summary_all.csv", index=False, encoding="utf-8-sig")
        print(f"\n💾 Summary đã lưu: retrieval_summary_all.csv")

        # In bảng so sánh với bài báo số 6
        print_comparison_table(summaries)
    else:
        print("\n❌ Không có kết quả nào để tổng hợp.")


if __name__ == "__main__":
    main()