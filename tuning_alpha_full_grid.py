"""Tune Adaptive Alpha on development data, then evaluate on an independent test split.

Run from the project root:
    python tuning_alpha_full_grid.py

Required project files:
- llm_chain.py
- vectordb.py
- KB1_Medical_Standard.xlsx
- KB2_Mom_Style.xlsx
- KB3_Information_Noise.xlsx
- an existing FAISS index configured by db_config.yml
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

from llm_chain import _get_production_hybrid_retriever
from vectordb import load_vector_db

RANDOM_SEED = 42
TOP_K = 5
DENSE_POOL_K = 50
BM25_POOL_K = 50
DEV_PER_PROFILE_PER_DATASET = 20
TEST_PER_PROFILE_PER_DATASET = 20
ALPHA_GRID = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]
TABLE_BONUS_GRID = [0.0, 0.05, 0.08, 0.10, 0.15, 0.20]
QUESTION_COLUMN = "Câu hỏi người dùng (Input)"
SOURCE_COLUMN = "Nguồn (Source)"
DATASET_FILES = {
    "KB1_Standard": "KB1_Medical_Standard.xlsx",
    "KB2_Mom_Style": "KB2_Mom_Style.xlsx",
    "KB3_Information_Noise": "KB3_Information_Noise.xlsx",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*")
PROFILE_ORDER = ["exact_lexical", "noisy_conversational", "quantitative", "semantic"]


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
    metadata = getattr(doc, "metadata", {}) or {}
    return normalize_source(metadata.get("source", ""))


def classify_query(question: str) -> str:
    """Keep the same profile rules as production llm_chain.py."""
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
    for dataset, filename in DATASET_FILES.items():
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        df = pd.read_excel(path)
        missing = {QUESTION_COLUMN, SOURCE_COLUMN} - set(df.columns)
        if missing:
            raise ValueError(f"{path} thiếu cột: {', '.join(sorted(missing))}")
        part = df[[QUESTION_COLUMN, SOURCE_COLUMN]].copy()
        part.columns = ["question", "source_raw"]
        part = part.dropna(subset=["question", "source_raw"])
        part["question"] = part["question"].astype(str).str.strip()
        part = part[part["question"].str.len() > 10]
        part["expected_sources"] = part["source_raw"].map(expected_sources)
        part = part[part["expected_sources"].map(bool)]
        part["profile"] = part["question"].map(classify_query)
        part["dataset"] = dataset
        frames.append(part)
    if not frames:
        raise ValueError("Không có dữ liệu hợp lệ.")
    return pd.concat(frames, ignore_index=True)


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_parts, test_parts = [], []
    for (_, _), group in df.groupby(["dataset", "profile"], sort=True):
        shuffled = group.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        n_dev = min(DEV_PER_PROFILE_PER_DATASET, max(1, len(shuffled) // 2))
        n_test = min(TEST_PER_PROFILE_PER_DATASET, len(shuffled) - n_dev)
        if n_test <= 0:
            continue
        dev_parts.append(shuffled.iloc[:n_dev])
        test_parts.append(shuffled.iloc[n_dev:n_dev + n_test])
    if not dev_parts or not test_parts:
        raise ValueError("Không thể tạo development/test split.")
    return pd.concat(dev_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


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

    def key(doc) -> str:
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


def build_cache(df: pd.DataFrame, db, cache, name: str):
    rows = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        rows.append((row, make_candidate_cache(row["question"], db, cache)))
        if idx % 25 == 0 or idx == total:
            print(f"  Cache {name}: {idx}/{total}")
    return rows


def tune_on_development(dev: pd.DataFrame, db, cache):
    tuning_rows, selected_rows = [], []
    cached = build_cache(dev, db, cache, "development")

    for profile in [p for p in PROFILE_ORDER if p in set(dev["profile"])]:
        print(f"\n🔧 Tuning profile: {profile}")
        profile_rows = [(row, cc) for row, cc in cached if row["profile"] == profile]
        best_record = None
        best_score = None

        for alpha in ALPHA_GRID:
            bonuses = TABLE_BONUS_GRID if profile == "quantitative" else [0.0]
            for bonus in bonuses:
                hits, rrs = [], []
                for row, candidate_cache in profile_rows:
                    docs = retrieve_from_cache(candidate_cache, alpha, profile, bonus)
                    hit, rr = metrics(docs, row["expected_sources"])
                    hits.append(hit)
                    rrs.append(rr)

                record = {
                    "profile": profile,
                    "alpha": alpha,
                    "table_bonus": bonus,
                    "n": len(profile_rows),
                    "hit_rate_at_5": float(np.mean(hits)),
                    "mrr_at_5": float(np.mean(rrs)),
                }
                tuning_rows.append(record)

                # Primary: MRR@5; secondary: Hit Rate@5.
                # Exact tie: prefer the smaller alpha and smaller bonus.
                score = (
                    record["mrr_at_5"],
                    record["hit_rate_at_5"],
                    -record["alpha"],
                    -record["table_bonus"],
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_record = record

        if best_record is None:
            raise RuntimeError(f"Không tìm được cấu hình cho {profile}")
        selected_rows.append(best_record)
        print(
            f"  ✅ alpha={best_record['alpha']:.1f} | "
            f"bonus={best_record['table_bonus']:.2f} | "
            f"Hit@5={best_record['hit_rate_at_5']:.4f} | "
            f"MRR@5={best_record['mrr_at_5']:.4f}"
        )

    tuning_df = pd.DataFrame(tuning_rows)
    selected_df = pd.DataFrame(selected_rows)
    alpha_map = {
        row["profile"]: float(row["alpha"])
        for _, row in selected_df.iterrows()
    }
    quantitative = selected_df[selected_df["profile"] == "quantitative"]
    table_bonus = float(quantitative.iloc[0]["table_bonus"]) if not quantitative.empty else 0.0
    return alpha_map, table_bonus, tuning_df, selected_df


def evaluate_cached(cached_rows, config_name: str, alpha_map=None, fixed_alpha=None, table_bonus=0.0):
    rows = []
    total = len(cached_rows)
    for idx, (row, candidate_cache) in enumerate(cached_rows, 1):
        profile = row["profile"]
        alpha = float(fixed_alpha) if fixed_alpha is not None else float(alpha_map[profile])
        start = time.perf_counter()
        docs = retrieve_from_cache(candidate_cache, alpha, profile, table_bonus)
        latency = time.perf_counter() - start
        hit, rr = metrics(docs, row["expected_sources"])
        rows.append({
            "split": "test",
            "config": config_name,
            "dataset": row["dataset"],
            "profile": profile,
            "question": row["question"],
            "alpha": alpha,
            "table_bonus": table_bonus,
            "hit_at_5": hit,
            "rr_at_5": rr,
            "latency_seconds": latency,
        })
        if idx % 50 == 0 or idx == total:
            print(f"  {config_name}: {idx}/{total}")
    return pd.DataFrame(rows)


def summarize_results(detailed: pd.DataFrame):
    by_profile = detailed.groupby(["config", "profile"], as_index=False).agg(
        n=("question", "count"),
        hit_rate_at_5=("hit_at_5", "mean"),
        mrr_at_5=("rr_at_5", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )
    overall = detailed.groupby("config", as_index=False).agg(
        n=("question", "count"),
        hit_rate_at_5=("hit_at_5", "mean"),
        mrr_at_5=("rr_at_5", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
    )
    return by_profile, overall


def plot_tuning(tuning_df: pd.DataFrame, selected_df: pd.DataFrame):
    for profile in PROFILE_ORDER:
        subset = tuning_df[tuning_df["profile"] == profile].copy()
        if subset.empty:
            continue
        subset = (
            subset.sort_values(
                ["alpha", "mrr_at_5", "hit_rate_at_5", "table_bonus"],
                ascending=[True, False, False, True],
            )
            .groupby("alpha", as_index=False)
            .first()
        )
        plt.figure(figsize=(9, 5.5))
        plt.plot(subset["alpha"], subset["mrr_at_5"], marker="o", label="MRR@5")
        plt.plot(subset["alpha"], subset["hit_rate_at_5"], marker="s", label="Hit Rate@5")
        selected = selected_df[selected_df["profile"] == profile]
        if not selected.empty:
            best_alpha = float(selected.iloc[0]["alpha"])
            plt.axvline(best_alpha, linestyle="--", label=f"Selected alpha={best_alpha:.1f}")
        plt.xticks(ALPHA_GRID)
        plt.xlabel("Alpha")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.title(f"Development tuning - {profile}")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"alpha_tuning_{profile}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_test(overall: pd.DataFrame):
    order = [f"Fixed alpha = {a:.1f}" for a in ALPHA_GRID] + ["Adaptive alpha tuned"]
    df = overall.set_index("config").reindex(order).dropna().reset_index()
    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))
    plt.plot(x, df["mrr_at_5"], marker="o", label="MRR@5")
    plt.plot(x, df["hit_rate_at_5"], marker="s", label="Hit Rate@5")
    plt.xticks(x, df["config"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Test comparison: Fixed Alpha and Tuned Adaptive Alpha")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaptive_alpha_full_grid_test_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def export_split(df: pd.DataFrame, filename: str):
    copy = df.copy()
    copy["expected_sources"] = copy["expected_sources"].map(
        lambda values: " | ".join(sorted(values))
    )
    copy.to_csv(filename, index=False, encoding="utf-8-sig")


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("📥 Đang đọc dữ liệu...")
    all_rows = load_all_rows()
    dev, test = make_split(all_rows)

    print("\n📌 Development:")
    print(dev.groupby(["dataset", "profile"]).size().to_string())
    print("\n📌 Test:")
    print(test.groupby(["dataset", "profile"]).size().to_string())

    export_split(dev, "adaptive_alpha_development_set.csv")
    export_split(test, "adaptive_alpha_test_set_full_grid.csv")

    print("\n⏳ Nạp FAISS và BM25...")
    db = load_vector_db()
    cache = _get_production_hybrid_retriever()

    print("\n🔧 Hiệu chỉnh trên development...")
    alpha_map, table_bonus, tuning_df, selected_df = tune_on_development(dev, db, cache)
    config = {**alpha_map, "table_bonus": table_bonus}

    with open("adaptive_alpha_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    tuning_df.to_csv("adaptive_alpha_full_grid_tuning.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv("adaptive_alpha_selected_config.csv", index=False, encoding="utf-8-sig")
    plot_tuning(tuning_df, selected_df)

    print("\n✅ Cấu hình đã khóa trên development:")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    print("\n🧱 Tạo cache cho test...")
    test_cache = build_cache(test, db, cache, "test")
    frames = []

    for alpha in ALPHA_GRID:
        name = f"Fixed alpha = {alpha:.1f}"
        print(f"\n🧪 {name}")
        frames.append(evaluate_cached(test_cache, name, fixed_alpha=alpha))

    print("\n🧪 Adaptive alpha tuned")
    frames.append(
        evaluate_cached(
            test_cache,
            "Adaptive alpha tuned",
            alpha_map=alpha_map,
            table_bonus=table_bonus,
        )
    )

    detailed = pd.concat(frames, ignore_index=True)
    by_profile, overall = summarize_results(detailed)
    detailed.to_csv("adaptive_alpha_full_grid_detailed.csv", index=False, encoding="utf-8-sig")
    by_profile.to_csv("adaptive_alpha_full_grid_by_profile.csv", index=False, encoding="utf-8-sig")
    overall.to_csv("adaptive_alpha_full_grid_overall.csv", index=False, encoding="utf-8-sig")
    plot_test(overall)

    print("\n📊 Kết quả test tổng hợp:")
    print(overall.to_string(index=False))
    print("\n✅ Hoàn tất. Kiểm tra adaptive_alpha_config.json và các file adaptive_alpha_full_grid_*.csv/png.")


if __name__ == "__main__":
    main()
