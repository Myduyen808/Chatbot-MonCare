"""
benchmark_adaptive_alpha.py
===========================
Đối chứng Fixed alpha và Adaptive alpha cho Hybrid Search của MomCare.

Mục tiêu:
- Cô lập ảnh hưởng của trọng số alpha.
- So sánh các alpha cố định với alpha thích nghi theo đúng logic production.
- Tách riêng ảnh hưởng của bonus ưu tiên chunk dạng bảng.

Kết quả sinh ra:
- adaptive_alpha_detailed.csv
- adaptive_alpha_summary.csv
- adaptive_alpha_statistics.csv
- adaptive_alpha_summary.json
- adaptive_alpha_hit_rate.png
- adaptive_alpha_mrr.png

Chạy tại thư mục gốc project:
    python benchmark_adaptive_alpha.py

Lưu ý:
- Script chỉ đánh giá retrieval, không gọi LLM, nên không tốn token API.
- VectorDB phải được xây dựng trước và metadata "source" phải chứa tên file nguồn.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_chain import _get_production_hybrid_retriever
from vectordb import load_vector_db


# =========================================================
# CẤU HÌNH
# =========================================================
RANDOM_SEED = 42
TOP_K = 5
CANDIDATE_K = 25
VECTOR_FETCH_K = 75
MAX_PER_GROUP_PER_DATASET = 30

QUESTION_COLUMN = "Câu hỏi người dùng (Input)"
SOURCE_COLUMN = "Nguồn (Source)"

DATASET_FILES = {
    "KB1_Standard": "KB1_Medical_Standard.xlsx",
    "KB2_Mom_Style": "KB2_Mom_Style.xlsx",
    "KB3_Information_Noise": "KB3_Information_Noise.xlsx",
}

# Giữ đúng regex của code production để kết quả đối chứng bám sát hệ thống.
QUANTITATIVE_PATTERN = re.compile(
    r"\d+\s*(?:mg|ml|g|kg|%|tháng|tuần|ngày|lần)",
    flags=re.IGNORECASE,
)

TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*"
)


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    fixed_alpha: Optional[float]
    use_table_bonus: bool = False


CONFIGS = [
    RetrievalConfig("Fixed alpha = 0.3", 0.3, False),
    RetrievalConfig("Fixed alpha = 0.4", 0.4, False),
    RetrievalConfig("Fixed alpha = 0.5", 0.5, False),
    RetrievalConfig("Fixed alpha = 0.7", 0.7, False),
    RetrievalConfig("Adaptive alpha", None, False),
    RetrievalConfig("Adaptive alpha + table bonus", None, True),
]


# =========================================================
# TIỆN ÍCH
# =========================================================
def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text)


def normalize_source_name(value: object) -> str:
    """Chuẩn hóa tên nguồn để so khớp metadata và nhãn Excel."""
    text = normalize_text(value).replace("\\", "/")
    return os.path.basename(text)


def split_expected_sources(value: object) -> set[str]:
    """Hỗ trợ một hoặc nhiều nguồn, phân tách bằng ;, |, xuống dòng hoặc dấu phẩy."""
    raw = str(value or "").strip()
    if not raw:
        return set()

    parts = re.split(r"[;|\n]+", raw)
    normalized = {normalize_source_name(part) for part in parts if part.strip()}
    return {item for item in normalized if item}


def document_source(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return normalize_source_name(metadata.get("source", ""))


def is_quantitative_query(question: str) -> bool:
    return bool(QUANTITATIVE_PATTERN.search(question.lower()))


def adaptive_alpha(question: str) -> float:
    # Bám đúng llm_chain.py: có số liệu -> 0.4, ngược lại -> 0.7.
    return 0.4 if is_quantitative_query(question) else 0.7


def reciprocal_rank_at_k(retrieved_sources: list[str], expected_sources: set[str], k: int) -> float:
    if not expected_sources:
        return 0.0
    for rank, source in enumerate(retrieved_sources[:k], start=1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_sources: list[str], expected_sources: set[str], k: int) -> int:
    return int(any(source in expected_sources for source in retrieved_sources[:k]))


# =========================================================
# NẠP TẬP KIỂM THỬ CÂN BẰNG
# =========================================================
def load_dataset(dataset_name: str, file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {file_path}. Hãy đặt script trong thư mục gốc project "
            "hoặc sửa DATASET_FILES."
        )

    df = pd.read_excel(path)
    missing = [column for column in (QUESTION_COLUMN, SOURCE_COLUMN) if column not in df.columns]
    if missing:
        raise ValueError(f"{file_path} thiếu cột: {missing}")

    subset = df[[QUESTION_COLUMN, SOURCE_COLUMN]].copy()
    subset.columns = ["question", "expected_source_raw"]
    subset = subset.dropna(subset=["question", "expected_source_raw"])
    subset["question"] = subset["question"].astype(str).str.strip()
    subset = subset[subset["question"].str.len() > 10]
    subset["expected_sources"] = subset["expected_source_raw"].apply(split_expected_sources)
    subset = subset[subset["expected_sources"].map(bool)]
    subset["query_type"] = np.where(
        subset["question"].map(is_quantitative_query),
        "Định lượng",
        "Ngữ nghĩa",
    )
    subset["dataset"] = dataset_name
    return subset.reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    samples = []
    for query_type, group in df.groupby("query_type", sort=False):
        n = min(max_per_group, len(group))
        samples.append(group.sample(n=n, random_state=seed))
    return pd.concat(samples, ignore_index=True)


def load_all_questions() -> pd.DataFrame:
    frames = []
    for dataset_name, file_path in DATASET_FILES.items():
        df = load_dataset(dataset_name, file_path)
        sampled = balanced_sample(df, MAX_PER_GROUP_PER_DATASET, RANDOM_SEED)
        frames.append(sampled)
        counts = sampled["query_type"].value_counts().to_dict()
        print(f"📘 {dataset_name}: {len(sampled)} câu {counts}")

    all_questions = pd.concat(frames, ignore_index=True)
    all_questions.insert(0, "question_id", [f"Q{i:04d}" for i in range(1, len(all_questions) + 1)])
    return all_questions


# =========================================================
# HYBRID SEARCH CÓ THỂ ĐIỀU KHIỂN ALPHA
# =========================================================
def retrieve_with_config(question: str, config: RetrievalConfig, db, cache):
    alpha = config.fixed_alpha if config.fixed_alpha is not None else adaptive_alpha(question)
    has_numbers = is_quantitative_query(question)

    vector_fetch_k = max(VECTOR_FETCH_K, CANDIDATE_K * 3)
    vector_docs = db.similarity_search(
        question,
        k=CANDIDATE_K,
        fetch_k=vector_fetch_k,
    )

    query_tokens = TOKEN_PATTERN.findall(question.lower())
    bm25_scores = cache["bm25"].get_scores(query_tokens)
    max_bm25 = float(max(bm25_scores)) if len(bm25_scores) else 0.0
    if max_bm25 <= 0:
        max_bm25 = 1.0

    combined = []
    for rank, doc in enumerate(vector_docs):
        vector_score = 1.0 / (rank + 1)
        bm25_index = cache["doc_to_index"].get(doc.page_content, -1)
        bm25_score = (
            float(bm25_scores[bm25_index]) / max_bm25
            if 0 <= bm25_index < len(bm25_scores)
            else 0.0
        )

        score = alpha * vector_score + (1.0 - alpha) * bm25_score

        # Chỉ cấu hình production mới cộng bonus; nhờ đó có thể tách ảnh hưởng alpha
        # khỏi ảnh hưởng ưu tiên chunk dạng bảng.
        if config.use_table_bonus and has_numbers and doc.metadata.get("chunk_type") == "data_table":
            score += 0.3

        combined.append((score, doc))

    combined.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in combined[:TOP_K]], alpha


# =========================================================
# CHẠY BENCHMARK
# =========================================================
def run_benchmark(questions: pd.DataFrame) -> pd.DataFrame:
    print("\n⏳ Đang nạp FAISS và BM25...")
    db = load_vector_db()
    cache = _get_production_hybrid_retriever()
    print("✅ Đã nạp xong bộ truy xuất.\n")

    rows = []
    total = len(questions) * len(CONFIGS)
    completed = 0

    for _, row in questions.iterrows():
        question = row["question"]
        expected_sources = row["expected_sources"]

        for config in CONFIGS:
            start = time.perf_counter()
            docs, selected_alpha = retrieve_with_config(question, config, db, cache)
            latency = time.perf_counter() - start

            sources = [document_source(doc) for doc in docs]
            hit5 = hit_at_k(sources, expected_sources, TOP_K)
            rr5 = reciprocal_rank_at_k(sources, expected_sources, TOP_K)

            rows.append(
                {
                    "question_id": row["question_id"],
                    "dataset": row["dataset"],
                    "query_type": row["query_type"],
                    "question": question,
                    "expected_sources": " | ".join(sorted(expected_sources)),
                    "config": config.name,
                    "selected_alpha": selected_alpha,
                    "table_bonus": config.use_table_bonus,
                    "hit_at_5": hit5,
                    "rr_at_5": rr5,
                    "latency_seconds": latency,
                    "retrieved_sources": " | ".join(sources),
                }
            )

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"📊 Tiến độ: {completed}/{total}")

    return pd.DataFrame(rows)


# =========================================================
# TỔNG HỢP, KIỂM ĐỊNH VÀ BIỂU ĐỒ
# =========================================================
def summarize_results(detailed: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detailed.groupby(["dataset", "query_type", "config"], as_index=False)
        .agg(
            n=("question_id", "count"),
            hit_rate_at_5=("hit_at_5", "mean"),
            mrr_at_5=("rr_at_5", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
            median_latency_seconds=("latency_seconds", "median"),
        )
    )

    overall = (
        detailed.groupby(["query_type", "config"], as_index=False)
        .agg(
            n=("question_id", "count"),
            hit_rate_at_5=("hit_at_5", "mean"),
            mrr_at_5=("rr_at_5", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
            median_latency_seconds=("latency_seconds", "median"),
        )
    )
    overall.insert(0, "dataset", "TỔNG HỢP")

    return pd.concat([summary, overall], ignore_index=True)


def run_paired_statistics(detailed: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon trên RR@5 theo từng câu hỏi; nếu scipy thiếu thì vẫn chạy benchmark."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("⚠️ Không có scipy, bỏ qua kiểm định Wilcoxon.")
        return pd.DataFrame()

    proposed = "Adaptive alpha + table bonus"
    baselines = ["Fixed alpha = 0.3", "Fixed alpha = 0.4", "Fixed alpha = 0.5", "Fixed alpha = 0.7"]
    rows = []

    for query_type in ["Định lượng", "Ngữ nghĩa", "Tất cả"]:
        subset = detailed if query_type == "Tất cả" else detailed[detailed["query_type"] == query_type]
        pivot = subset.pivot(index="question_id", columns="config", values="rr_at_5")

        for baseline in baselines:
            pair = pivot[[baseline, proposed]].dropna()
            differences = pair[proposed] - pair[baseline]

            if pair.empty or np.allclose(differences.to_numpy(), 0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    pair[proposed],
                    pair[baseline],
                    zero_method="wilcox",
                    alternative="two-sided",
                )

            rows.append(
                {
                    "query_type": query_type,
                    "baseline": baseline,
                    "proposed": proposed,
                    "n_pairs": len(pair),
                    "baseline_mean_rr": pair[baseline].mean(),
                    "proposed_mean_rr": pair[proposed].mean(),
                    "median_difference": differences.median(),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


def plot_overall(summary: pd.DataFrame) -> None:
    overall = summary[summary["dataset"] == "TỔNG HỢP"].copy()
    config_order = [config.name for config in CONFIGS]

    for metric, ylabel, filename in [
        ("hit_rate_at_5", "Hit Rate@5", "adaptive_alpha_hit_rate.png"),
        ("mrr_at_5", "MRR@5", "adaptive_alpha_mrr.png"),
    ]:
        pivot = overall.pivot(index="config", columns="query_type", values=metric).reindex(config_order)
        ax = pivot.plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"So sánh {ylabel} giữa trọng số cố định và trọng số thích nghi")
        ax.set_xlabel("Cấu hình")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(title="Nhóm truy vấn")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    questions = load_all_questions()
    questions_for_export = questions.copy()
    questions_for_export["expected_sources"] = questions_for_export["expected_sources"].apply(
        lambda values: " | ".join(sorted(values))
    )
    questions_for_export.to_csv("adaptive_alpha_test_set.csv", index=False, encoding="utf-8-sig")

    print("\n📌 Phân bố tập kiểm thử:")
    print(questions.groupby(["dataset", "query_type"]).size().to_string())

    detailed = run_benchmark(questions)
    summary = summarize_results(detailed)
    statistics = run_paired_statistics(detailed)

    detailed.to_csv("adaptive_alpha_detailed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("adaptive_alpha_summary.csv", index=False, encoding="utf-8-sig")
    if not statistics.empty:
        statistics.to_csv("adaptive_alpha_statistics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "settings": {
            "random_seed": RANDOM_SEED,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "max_per_group_per_dataset": MAX_PER_GROUP_PER_DATASET,
            "production_rule": "alpha=0.4 nếu có số liệu/đơn vị; ngược lại alpha=0.7; table bonus=0.3",
        },
        "summary": summary.to_dict(orient="records"),
        "statistics": statistics.to_dict(orient="records") if not statistics.empty else [],
    }
    with open("adaptive_alpha_summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    plot_overall(summary)

    print("\n✅ Hoàn tất. Các file đã tạo:")
    for filename in [
        "adaptive_alpha_test_set.csv",
        "adaptive_alpha_detailed.csv",
        "adaptive_alpha_summary.csv",
        "adaptive_alpha_statistics.csv",
        "adaptive_alpha_summary.json",
        "adaptive_alpha_hit_rate.png",
        "adaptive_alpha_mrr.png",
    ]:
        if Path(filename).exists():
            print(f"   - {filename}")

    print("\n📊 Kết quả tổng hợp:")
    print(
        summary[summary["dataset"] == "TỔNG HỢP"]
        [["query_type", "config", "n", "hit_rate_at_5", "mrr_at_5", "mean_latency_seconds"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
