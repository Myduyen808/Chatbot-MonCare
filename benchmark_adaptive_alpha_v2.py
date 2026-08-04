"""
Đánh giá Adaptive Weighting v2 với tách development/test.

- Development: chọn alpha tốt nhất riêng cho từng kiểu truy vấn.
- Test: so sánh cấu hình thích nghi đã khóa với các alpha cố định.
- Hai danh sách ứng viên FAISS và BM25 được tạo độc lập rồi hợp nhất.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from llm_chain import _get_production_hybrid_retriever
from vectordb import load_vector_db

RANDOM_SEED = 42
TOP_K = 5
DENSE_POOL_K = 50
BM25_POOL_K = 50
DEV_PER_PROFILE_PER_DATASET = 20
TEST_PER_PROFILE_PER_DATASET = 20
ALPHA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
TABLE_BONUS_GRID = [0.0, 0.05, 0.08, 0.10]

QUESTION_COLUMN = "Câu hỏi người dùng (Input)"
SOURCE_COLUMN = "Nguồn (Source)"
DATASET_FILES = {
    "KB1_Standard": "KB1_Medical_Standard.xlsx",
    "KB2_Mom_Style": "KB2_Mom_Style.xlsx",
    "KB3_Information_Noise": "KB3_Information_Noise.xlsx",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*")


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    return re.sub(r"\s+", " ", text)


def normalize_source(value: object) -> str:
    return os.path.basename(normalize_text(value).replace("\\", "/"))


def expected_sources(value: object) -> set[str]:
    return {
        normalize_source(part)
        for part in re.split(r"[;|\n]+", str(value or ""))
        if part.strip()
    }


def document_source(doc) -> str:
    return normalize_source((getattr(doc, "metadata", {}) or {}).get("source", ""))


def classify_query(question: str) -> str:
    q = normalize_text(question)
    quantitative_pattern = re.compile(
        r"(?:\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml|g|kg|%|iu|kcal|"
        r"tháng|tuần|ngày|giờ|phút|lần|tuổi)\b)|"
        r"(?:bao nhiêu|mấy lần|mỗi ngày|mỗi tuần|liều|tần suất)",
        flags=re.IGNORECASE,
    )
    if quantitative_pattern.search(q):
        return "quantitative"

    exact_terms = [
        "vitamin d", "paracetamol", "ibuprofen", "amoxicillin", "oxytocin",
        "aspirin", "sắt", "canxi", "axit folic", "tắc tia sữa",
        "viêm tuyến vú", "băng huyết", "sản dịch", "vàng da",
        "tưa miệng", "ăn dặm", "bú mẹ",
    ]
    if any(term in q for term in exact_terms):
        return "exact_lexical"

    noisy_markers = [
        "mom", "mẹ ơi", "bé nhà em", "bé nhà mình", "ạ", "nha", "nhỉ",
        "kiểu", "sao á", "vậy ta", "hông", "hong", "ko ", "k ", "mik",
        "mn", " z", "rồi á",
    ]
    if any(marker in q for marker in noisy_markers):
        return "noisy_conversational"
    return "semantic"


def load_all_rows() -> pd.DataFrame:
    frames = []
    for dataset, path_str in DATASET_FILES.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy {path}")
        df = pd.read_excel(path)
        required = [QUESTION_COLUMN, SOURCE_COLUMN]
        if any(col not in df.columns for col in required):
            raise ValueError(f"{path} thiếu cột bắt buộc")
        part = df[required].copy()
        part.columns = ["question", "source_raw"]
        part = part.dropna()
        part["question"] = part["question"].astype(str).str.strip()
        part = part[part["question"].str.len() > 10]
        part["expected_sources"] = part["source_raw"].map(expected_sources)
        part = part[part["expected_sources"].map(bool)]
        part["profile"] = part["question"].map(classify_query)
        part["dataset"] = dataset
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_parts, test_parts = [], []
    for (dataset, profile), group in df.groupby(["dataset", "profile"]):
        shuffled = group.sample(frac=1, random_state=RANDOM_SEED)
        n_dev = min(DEV_PER_PROFILE_PER_DATASET, max(1, len(shuffled) // 2))
        remaining = len(shuffled) - n_dev
        n_test = min(TEST_PER_PROFILE_PER_DATASET, remaining)
        if n_test == 0:
            continue
        dev_parts.append(shuffled.iloc[:n_dev])
        test_parts.append(shuffled.iloc[n_dev:n_dev + n_test])
    dev = pd.concat(dev_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return dev, test


def make_candidate_cache(question: str, db, cache):
    dense_docs = db.similarity_search(
        question,
        k=DENSE_POOL_K,
        fetch_k=max(DENSE_POOL_K * 3, 150),
    )
    query_tokens = TOKEN_PATTERN.findall(question.lower())
    bm25_scores = cache["bm25"].get_scores(query_tokens)
    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda idx: float(bm25_scores[idx]),
        reverse=True,
    )[:BM25_POOL_K]
    bm25_docs = [cache["valid_docs"][idx] for idx in top_indices]

    def key(doc):
        return re.sub(r"\s+", " ", str(doc.page_content)).strip()[:1000]

    dense_rank = {key(doc): rank for rank, doc in enumerate(dense_docs, 1)}
    bm25_rank = {key(doc): rank for rank, doc in enumerate(bm25_docs, 1)}
    candidates = {}
    for doc in dense_docs + bm25_docs:
        candidates.setdefault(key(doc), doc)
    return candidates, dense_rank, bm25_rank


def retrieve_from_cache(candidate_cache, alpha: float, profile: str, table_bonus: float):
    candidates, dense_rank, bm25_rank = candidate_cache
    ranked = []
    for key, doc in candidates.items():
        dense_score = 1.0 / dense_rank[key] if key in dense_rank else 0.0
        lexical_score = 1.0 / bm25_rank[key] if key in bm25_rank else 0.0
        score = alpha * dense_score + (1.0 - alpha) * lexical_score
        if profile == "quantitative" and doc.metadata.get("chunk_type") == "data_table":
            score += table_bonus
        ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_K]]


def metrics(docs, expected: set[str]) -> tuple[int, float]:
    sources = [document_source(doc) for doc in docs]
    for rank, source in enumerate(sources, 1):
        if source in expected:
            return 1, 1.0 / rank
    return 0, 0.0


def evaluate(df: pd.DataFrame, alpha_map: dict[str, float] | None, fixed_alpha: float | None,
             table_bonus: float, db, cache, split_name: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    for idx, row in df.reset_index(drop=True).iterrows():
        profile = row["profile"]
        alpha = fixed_alpha if fixed_alpha is not None else alpha_map[profile]
        start = time.perf_counter()
        candidate_cache = make_candidate_cache(row["question"], db, cache)
        docs = retrieve_from_cache(candidate_cache, alpha, profile, table_bonus)
        latency = time.perf_counter() - start
        hit, rr = metrics(docs, row["expected_sources"])
        rows.append({
            "split": split_name,
            "dataset": row["dataset"],
            "profile": profile,
            "question": row["question"],
            "alpha": alpha,
            "table_bonus": table_bonus,
            "hit_at_5": hit,
            "rr_at_5": rr,
            "latency_seconds": latency,
        })
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  {split_name}: {idx + 1}/{total}")
    return pd.DataFrame(rows)


def tune_on_development(dev: pd.DataFrame, db, cache):
    tuning_rows = []
    best_map = {}

    # Cache ứng viên để không truy xuất lại cho mỗi alpha.
    cached = []
    for _, row in dev.iterrows():
        cached.append((row, make_candidate_cache(row["question"], db, cache)))

    for profile in sorted(dev["profile"].unique()):
        profile_rows = [(row, cc) for row, cc in cached if row["profile"] == profile]
        best = None
        for alpha in ALPHA_GRID:
            bonuses = TABLE_BONUS_GRID if profile == "quantitative" else [0.0]
            for bonus in bonuses:
                hits, rrs = [], []
                for row, cc in profile_rows:
                    docs = retrieve_from_cache(cc, alpha, profile, bonus)
                    hit, rr = metrics(docs, row["expected_sources"])
                    hits.append(hit); rrs.append(rr)
                record = {
                    "profile": profile,
                    "alpha": alpha,
                    "table_bonus": bonus,
                    "n": len(profile_rows),
                    "hit_rate_at_5": float(np.mean(hits)),
                    "mrr_at_5": float(np.mean(rrs)),
                }
                tuning_rows.append(record)
                score = (record["mrr_at_5"], record["hit_rate_at_5"], -alpha)
                if best is None or score > best[0]:
                    best = (score, record)
        best_map[profile] = best[1]["alpha"]

    tuning_df = pd.DataFrame(tuning_rows)
    quantitative_best = tuning_df[tuning_df["profile"] == "quantitative"].sort_values(
        ["mrr_at_5", "hit_rate_at_5"], ascending=False
    ).iloc[0]
    table_bonus = float(quantitative_best["table_bonus"])
    return best_map, table_bonus, tuning_df


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["config", "profile"], as_index=False).agg(
        n=("question", "count"),
        hit_rate_at_5=("hit_at_5", "mean"),
        mrr_at_5=("rr_at_5", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )


def main():
    random.seed(RANDOM_SEED)
    all_rows = load_all_rows()
    dev, test = make_split(all_rows)
    print("📌 Development:", dev.groupby(["dataset", "profile"]).size().to_dict())
    print("📌 Test:", test.groupby(["dataset", "profile"]).size().to_dict())

    print("⏳ Nạp FAISS và BM25...")
    db = load_vector_db()
    cache = _get_production_hybrid_retriever()

    print("🔧 Hiệu chỉnh alpha trên development...")
    alpha_map, table_bonus, tuning_df = tune_on_development(dev, db, cache)
    config = {**alpha_map, "table_bonus": table_bonus}
    with open("adaptive_alpha_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    tuning_df.to_csv("adaptive_alpha_v2_tuning.csv", index=False, encoding="utf-8-sig")
    print("✅ Cấu hình đã khóa:", config)

    result_frames = []
    for alpha in [0.3, 0.4, 0.5, 0.7]:
        print(f"🧪 Test Fixed alpha={alpha}")
        part = evaluate(test, None, alpha, 0.0, db, cache, "test")
        part["config"] = f"Fixed alpha = {alpha}"
        result_frames.append(part)

    print("🧪 Test Adaptive v2")
    adaptive = evaluate(test, alpha_map, None, table_bonus, db, cache, "test")
    adaptive["config"] = "Adaptive alpha v2"
    result_frames.append(adaptive)

    # =========================================================
    # KIỂM ĐỊNH WILCOXON THEO TỪNG CÂU HỎI
    # =========================================================

    statistics_rows = []

    pair_keys = [
        "dataset",
        "profile",
        "question",
    ]

    adaptive_rows = (
        detailed[
            detailed["config"] == "Adaptive alpha v2"
        ][
            pair_keys + ["hit_at_5", "rr_at_5"]
        ]
        .rename(
            columns={
                "hit_at_5": "adaptive_hit",
                "rr_at_5": "adaptive_rr",
            }
        )
    )

    for fixed_config in [
        "Fixed alpha = 0.3",
        "Fixed alpha = 0.4",
        "Fixed alpha = 0.5",
        "Fixed alpha = 0.7",
    ]:
        fixed_rows = (
            detailed[
                detailed["config"] == fixed_config
            ][
                pair_keys + ["hit_at_5", "rr_at_5"]
            ]
            .rename(
                columns={
                    "hit_at_5": "fixed_hit",
                    "rr_at_5": "fixed_rr",
                }
            )
        )

        paired = adaptive_rows.merge(
            fixed_rows,
            on=pair_keys,
            how="inner",
            validate="one_to_one",
        )

        if paired.empty:
            print(
                f"⚠️ Không có dữ liệu ghép cặp cho {fixed_config}."
            )
            continue

        # Wilcoxon không xử lý được trường hợp mọi chênh lệch đều bằng 0.
        rr_difference = (
            paired["adaptive_rr"] - paired["fixed_rr"]
        )

        hit_difference = (
            paired["adaptive_hit"] - paired["fixed_hit"]
        )

        if (rr_difference != 0).any():
            rr_test = wilcoxon(
                paired["adaptive_rr"],
                paired["fixed_rr"],
                zero_method="pratt",
                alternative="two-sided",
            )

            rr_statistic = float(rr_test.statistic)
            rr_p_value = float(rr_test.pvalue)
        else:
            rr_statistic = 0.0
            rr_p_value = 1.0

        if (hit_difference != 0).any():
            hit_test = wilcoxon(
                paired["adaptive_hit"],
                paired["fixed_hit"],
                zero_method="pratt",
                alternative="two-sided",
            )

            hit_statistic = float(hit_test.statistic)
            hit_p_value = float(hit_test.pvalue)
        else:
            hit_statistic = 0.0
            hit_p_value = 1.0

        statistics_rows.append(
            {
                "comparison": (
                    f"Adaptive alpha v2 vs {fixed_config}"
                ),
                "metric": "MRR@5",
                "n_pairs": len(paired),
                "adaptive_mean": paired["adaptive_rr"].mean(),
                "fixed_mean": paired["fixed_rr"].mean(),
                "mean_difference": rr_difference.mean(),
                "better_cases": int((rr_difference > 0).sum()),
                "equal_cases": int((rr_difference == 0).sum()),
                "worse_cases": int((rr_difference < 0).sum()),
                "statistic": rr_statistic,
                "p_value": rr_p_value,
                "significant_0_05": rr_p_value < 0.05,
            }
        )

        statistics_rows.append(
            {
                "comparison": (
                    f"Adaptive alpha v2 vs {fixed_config}"
                ),
                "metric": "Hit Rate@5",
                "n_pairs": len(paired),
                "adaptive_mean": paired["adaptive_hit"].mean(),
                "fixed_mean": paired["fixed_hit"].mean(),
                "mean_difference": hit_difference.mean(),
                "better_cases": int((hit_difference > 0).sum()),
                "equal_cases": int((hit_difference == 0).sum()),
                "worse_cases": int((hit_difference < 0).sum()),
                "statistic": hit_statistic,
                "p_value": hit_p_value,
                "significant_0_05": hit_p_value < 0.05,
            }
        )

    statistics_df = pd.DataFrame(statistics_rows)

    statistics_df.to_csv(
        "adaptive_alpha_v2_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = detailed.groupby("config", as_index=False).agg(
        n=("question", "count"),
        hit_rate_at_5=("hit_at_5", "mean"),
        mrr_at_5=("rr_at_5", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )
    overall.to_csv("adaptive_alpha_v2_overall.csv", index=False, encoding="utf-8-sig")

    # Biểu đồ tổng hợp.
    x = np.arange(len(overall))
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(x, overall["mrr_at_5"])
    plt.xticks(x, overall["config"], rotation=20, ha="right")
    plt.ylabel("MRR@5")
    plt.title("So sánh MRR@5 trên tập kiểm thử độc lập")
    for bar, value in zip(bars, overall["mrr_at_5"]):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.005, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig("adaptive_alpha_v2_mrr.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n📊 Kết quả test tổng hợp:")
    print(overall.to_string(index=False))
    print(
    "\n✅ Đã tạo adaptive_alpha_config.json, các file kết quả v2 và adaptive_alpha_v2_statistics.csv."
)


if __name__ == "__main__":
    main()
