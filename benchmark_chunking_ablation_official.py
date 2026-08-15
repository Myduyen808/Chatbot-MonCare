#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_chunking_ablation_official.py
=======================================

Chunking ablation cho MomCare trên ĐÚNG official ViMedAQA test split.

Thiết lập bám theo evaluation/vimedaqa_clean_benchmark.py:
- Dataset: tmnam20/ViMedAQA
- Config: all
- Split: test
- Official test: 2,217 mẫu
- Loại các mẫu thiếu question/answer/context giống clean benchmark
- Kỳ vọng còn 2,213 mẫu hợp lệ
- Chỉ `context` được lập chỉ mục
- `question` chỉ dùng làm query
- `answer` không đưa vào index
- Embedding: keepitreal/vietnamese-sbert
- Retrieval: Adaptive Hybrid FAISS + BM25
- Dense pool = 50, BM25 pool = 50, candidate = 25, final K = 5

Chunking configs:
    512 / 100
    1000 / 200
    1800 / 360   <- production
    3000 / 600

Không gọi Groq/LLM.
Không đụng vào VectorDB production.

Chạy smoke test:
    python benchmark_chunking_ablation_official.py --limit 50

Chạy full:
    python benchmark_chunking_ablation_official.py

Có thể chạy từng config:
    python benchmark_chunking_ablation_official.py --configs "512,100"
    python benchmark_chunking_ablation_official.py --configs "1000,200"
    python benchmark_chunking_ablation_official.py --configs "1800,360"
    python benchmark_chunking_ablation_official.py --configs "3000,600"

Kết quả:
    chunking_ablation_official_results/
        clean_test_rows.csv
        excluded_samples.csv
        details_*.csv
        summary.csv
        summary_by_topic.csv
        experimental_index_*/
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CẤU HÌNH GIỐNG CLEAN BENCHMARK
# ============================================================

DATASET_NAME = "tmnam20/ViMedAQA"
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
EXPECTED_TEST_SAMPLES = 2217
EXPECTED_VALID_SAMPLES = 2213

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

DENSE_POOL_K = 50
BM25_POOL_K = 50
CANDIDATE_K = 25
FINAL_K = 5

ALPHA_BY_PROFILE = {
    "exact_lexical": 0.20,
    "semantic": 0.30,
    "noisy_conversational": 0.30,
    "quantitative": 0.40,
}

TOPIC_MAP = {
    0: "body-part",
    1: "disease",
    2: "drug",
    3: "medicine",
}

CHUNK_CONFIGS = [
    (512, 100),
    (1000, 200),
    (1800, 360),
    (3000, 600),
]

TOKEN_RE = re.compile(
    r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*"
)


# ============================================================
# HÀM GIỐNG CLEAN BENCHMARK
# ============================================================

def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text).lower())


def context_hash(text: object) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def classify_query(question: str) -> str:
    """Mirror MomCare final Adaptive Alpha query profiles."""
    q = normalize_text(question).lower()

    quantitative_markers = (
        "bao nhiêu", "mấy lần", "mấy bữa", "mấy ngày", "mấy tháng",
        "mấy tuần", "bao lâu", "mỗi ngày", "mỗi tuần", "mỗi lần",
        "liều", "liều lượng", "tần suất", "số lượng", "lượng bao nhiêu",
    )
    if any(marker in q for marker in quantitative_markers):
        return "quantitative"

    measurement_pattern = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml|g|kg|%|iu|kcal)\b",
        flags=re.IGNORECASE,
    )
    if measurement_pattern.search(q):
        return "quantitative"

    exact_terms = (
        "vitamin", "vitamin d", "paracetamol", "ibuprofen", "amoxicillin",
        "oxytocin", "aspirin", "sắt", "canxi", "axit folic", "tắc tia sữa",
        "viêm tuyến vú", "băng huyết", "sản dịch", "vàng da", "tưa miệng",
        "ăn dặm", "bú mẹ", "sữa mẹ",
    )
    if any(term in q for term in exact_terms):
        return "exact_lexical"

    noisy_markers = (
        "mom", "mẹ ơi", "bé nhà em", "bé nhà mình", "ạ", "nha", "nhỉ",
        "kiểu", "sao á", "vậy ta", "hông", "hong", "ko ", "k ", "mik",
        "mn", "rồi á",
    )
    if any(marker in q for marker in noisy_markers):
        return "noisy_conversational"

    return "semantic"


def expand_retrieval_query(question: str) -> str:
    """Mirror clean benchmark's retrieval-only alias expansion."""
    original = normalize_text(question)
    lowered = original.lower()
    additions: list[str] = []

    if "ăn dặm" in lowered and "ăn bổ sung" not in lowered:
        additions.append("ăn bổ sung")

    if "ăn dặm" in lowered and any(
        marker in lowered
        for marker in ("có nên", "bắt đầu", "khi nào", "từ mấy tháng")
    ):
        additions.append("thời điểm bắt đầu ăn bổ sung")

    if any(
        marker in lowered
        for marker in ("mấy bữa", "bao nhiêu bữa", "số bữa")
    ):
        additions.append("tần suất ăn bổ sung")

    return normalize_text(" ".join([original, *additions]))


# ============================================================
# LOAD ĐÚNG OFFICIAL TEST
# ============================================================

def load_official_clean_test(
    output_dir: Path,
    limit: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    print(
        f"🌐 Loading Hugging Face dataset: "
        f"{DATASET_NAME} / {DATASET_CONFIG} / {DATASET_SPLIT}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )
    df = dataset.to_pandas().reset_index(drop=True)

    required = {"question_idx", "question", "answer", "context", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"ViMedAQA thiếu các cột bắt buộc: {sorted(missing)}"
        )

    if len(df) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Sai số mẫu official test: nhận {len(df)}, "
            f"kỳ vọng {EXPECTED_TEST_SAMPLES}."
        )

    valid_rows = []
    excluded_rows = []

    for row_index, row in df.iterrows():
        question = normalize_text(row["question"])
        answer = normalize_text(row["answer"])
        context = normalize_text(row["context"])

        missing_fields = [
            name
            for name, value in (
                ("question", question),
                ("answer", answer),
                ("context", context),
            )
            if not value
        ]

        if missing_fields:
            excluded_rows.append({
                "row_index": int(row_index),
                "question_idx": row.get("question_idx", ""),
                "missing_fields": "|".join(missing_fields),
            })
            continue

        topic_raw = row["topic"]
        try:
            topic_num = int(float(topic_raw))
            topic = TOPIC_MAP.get(topic_num, str(topic_num))
        except Exception:
            topic = normalize_text(topic_raw) or "unknown"

        valid_rows.append({
            "row_index": int(row_index),
            "sample_id": str(row.get("question_idx", row_index)),
            "topic": topic,
            "question": question,
            "reference_answer": answer,
            "context": context,
            "gold_context_hash": context_hash(context),
        })

    clean = pd.DataFrame(valid_rows)
    excluded = pd.DataFrame(excluded_rows)

    if len(clean) != EXPECTED_VALID_SAMPLES:
        raise RuntimeError(
            f"Số mẫu clean khác benchmark gốc: nhận {len(clean)}, "
            f"kỳ vọng {EXPECTED_VALID_SAMPLES}. "
            f"Excluded={len(excluded)}."
        )

    clean.to_csv(
        output_dir / "clean_test_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        output_dir / "excluded_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✅ Official test samples : {len(df)}")
    print(f"✅ Valid clean samples   : {len(clean)}")
    print(f"✅ Excluded samples      : {len(excluded)}")
    print("✅ Indexed field         : context ONLY")
    print("✅ Question/Answer       : NOT indexed")

    if limit and 0 < limit < len(clean):
        # Smoke test: lấy mẫu cân bằng theo topic nếu có thể.
        parts = []
        groups = list(clean.groupby("topic", sort=True))
        per_group = max(1, limit // max(1, len(groups)))

        for _, group in groups:
            n = min(per_group, len(group))
            parts.append(group.sample(n=n, random_state=seed))

        sampled = pd.concat(parts, ignore_index=False)

        if len(sampled) < limit:
            remaining = clean.drop(index=sampled.index, errors="ignore")
            extra_n = min(limit - len(sampled), len(remaining))
            if extra_n > 0:
                sampled = pd.concat([
                    sampled,
                    remaining.sample(n=extra_n, random_state=seed),
                ])

        clean = (
            sampled
            .head(limit)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        print(f"🧪 Smoke-test subset     : {len(clean)}")

    clean = clean.reset_index(drop=True)
    clean.insert(
        0,
        "question_id",
        [f"Q{i:04d}" for i in range(1, len(clean) + 1)],
    )

    print("\n📌 Topic distribution:")
    print(clean["topic"].value_counts().to_string())
    print()

    return clean


# ============================================================
# CHUNKING / INDEX
# ============================================================

def build_documents(
    dataset: pd.DataFrame,
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(overlap),
    )

    docs: list[Document] = []

    # QUAN TRỌNG:
    # Không deduplicate context. Benchmark gốc cũng đi từng row official test.
    for _, row in dataset.iterrows():
        context = row["context"]

        chunks = splitter.split_text(context)
        if not chunks:
            chunks = [context]

        for chunk_idx, chunk in enumerate(chunks):
            chunk = normalize_text(chunk)
            if not chunk:
                continue

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "split": "test",
                        "row_index": int(row["row_index"]),
                        "sample_id": str(row["sample_id"]),
                        "context_hash": str(row["gold_context_hash"]),
                        "chunk_in_context": int(chunk_idx),
                        "topic": str(row["topic"]),
                        "content_field": "context_only",
                    },
                )
            )

    return docs


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"🧠 Embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_and_bm25(
    docs: list[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: Path,
):
    if not docs:
        raise RuntimeError("Không tạo được context chunk nào.")

    print(f"  🧱 Building FAISS from {len(docs):,} chunks...")
    db = FAISS.from_documents(docs, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))

    print("  🔤 Building BM25...")
    bm25_docs: list[Document] = []
    corpus: list[list[str]] = []

    for doc in docs:
        toks = tokenize(doc.page_content)
        if not toks:
            continue
        bm25_docs.append(doc)
        corpus.append(toks)

    bm25 = BM25Okapi(corpus)

    return db, bm25, bm25_docs


# ============================================================
# ADAPTIVE HYBRID RETRIEVAL
# ============================================================

def doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return (
        f"{md.get('context_hash', '')}|"
        f"{md.get('row_index', '')}|"
        f"{md.get('chunk_in_context', '')}|"
        f"{normalize_text(doc.page_content)[:1000]}"
    )


def hybrid_search(
    question: str,
    db: FAISS,
    bm25: BM25Okapi,
    bm25_docs: list[Document],
) -> tuple[list[Document], str, float]:
    profile = classify_query(question)
    alpha = float(ALPHA_BY_PROFILE[profile])
    query = expand_retrieval_query(question)

    # Dense candidates
    try:
        dense_docs = db.similarity_search(
            query,
            k=DENSE_POOL_K,
            fetch_k=max(DENSE_POOL_K * 3, 150),
        )
    except TypeError:
        dense_docs = db.similarity_search(
            query,
            k=DENSE_POOL_K,
        )

    # BM25 candidates
    q_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_indices = np.argsort(bm25_scores)[::-1][:BM25_POOL_K]
    lexical_docs = [
        bm25_docs[int(i)]
        for i in bm25_indices
        if 0 <= int(i) < len(bm25_docs)
    ]

    dense_rank = {
        doc_key(doc): rank
        for rank, doc in enumerate(dense_docs, start=1)
    }
    lexical_rank = {
        doc_key(doc): rank
        for rank, doc in enumerate(lexical_docs, start=1)
    }

    candidates: dict[str, Document] = {}
    for doc in dense_docs + lexical_docs:
        candidates.setdefault(doc_key(doc), doc)

    scored: list[tuple[float, Document]] = []

    for key, doc in candidates.items():
        dense_score = (
            1.0 / dense_rank[key]
            if key in dense_rank
            else 0.0
        )
        lexical_score = (
            1.0 / lexical_rank[key]
            if key in lexical_rank
            else 0.0
        )

        score = (
            alpha * dense_score
            + (1.0 - alpha) * lexical_score
        )
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)

    # Giống scale benchmark: merge -> candidate 25 -> final 5.
    candidate_docs = [
        doc for _, doc in scored[:CANDIDATE_K]
    ]
    return candidate_docs[:FINAL_K], profile, alpha


# ============================================================
# EVALUATION
# ============================================================

def gold_rank(
    retrieved_docs: list[Document],
    gold_hash: str,
) -> int | None:
    for rank, doc in enumerate(retrieved_docs[:FINAL_K], start=1):
        if str((doc.metadata or {}).get("context_hash", "")) == str(gold_hash):
            return rank
    return None


def evaluate_config(
    dataset: pd.DataFrame,
    db: FAISS,
    bm25: BM25Okapi,
    bm25_docs: list[Document],
    chunk_size: int,
    overlap: int,
    detail_path: Path,
) -> pd.DataFrame:
    rows = []
    total = len(dataset)

    for idx, row in dataset.iterrows():
        started = time.perf_counter()

        retrieved, profile, alpha = hybrid_search(
            question=row["question"],
            db=db,
            bm25=bm25,
            bm25_docs=bm25_docs,
        )

        latency = time.perf_counter() - started
        rank = gold_rank(
            retrieved,
            row["gold_context_hash"],
        )

        rows.append({
            "question_id": row["question_id"],
            "row_index": int(row["row_index"]),
            "sample_id": row["sample_id"],
            "topic": row["topic"],
            "question": row["question"],
            "query_profile": profile,
            "selected_alpha": alpha,
            "gold_context_hash": row["gold_context_hash"],
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(overlap),
            "gold_rank": rank if rank is not None else 0,
            "hit_at_1": int(rank == 1),
            "hit_at_3": int(rank is not None and rank <= 3),
            "hit_at_5": int(rank is not None and rank <= 5),
            "rr_at_5": (1.0 / rank) if rank is not None else 0.0,
            "latency_seconds": latency,
            "retrieved_context_hashes": "|".join(
                str((doc.metadata or {}).get("context_hash", ""))
                for doc in retrieved
            ),
        })

        current = len(rows)

        if current % 100 == 0 or current == total:
            temp = pd.DataFrame(rows)
            temp.to_csv(
                detail_path,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"    [{current:>4}/{total}] "
                f"Hit@1={temp['hit_at_1'].mean():.4f} | "
                f"Hit@3={temp['hit_at_3'].mean():.4f} | "
                f"Hit@5={temp['hit_at_5'].mean():.4f} | "
                f"MRR@5={temp['rr_at_5'].mean():.4f}"
            )

    details = pd.DataFrame(rows)
    details.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )
    return details


def summarize_config(
    details: pd.DataFrame,
    chunk_count: int,
    n_source_rows: int,
    build_seconds: float,
) -> dict:
    return {
        "chunk_size": int(details["chunk_size"].iloc[0]),
        "chunk_overlap": int(details["chunk_overlap"].iloc[0]),
        "n_queries": int(len(details)),
        "source_rows_indexed": int(n_source_rows),
        "total_chunks": int(chunk_count),
        "avg_chunks_per_source_row": float(
            chunk_count / max(1, n_source_rows)
        ),
        "hit_at_1": float(details["hit_at_1"].mean()),
        "hit_at_3": float(details["hit_at_3"].mean()),
        "hit_at_5": float(details["hit_at_5"].mean()),
        "mrr_at_5": float(details["rr_at_5"].mean()),
        "mean_latency_seconds": float(
            details["latency_seconds"].mean()
        ),
        "median_latency_seconds": float(
            details["latency_seconds"].median()
        ),
        "build_seconds": float(build_seconds),
    }


def rebuild_summary_from_details(
    output_dir: Path,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    topic_frames = []

    for chunk_size, overlap in CHUNK_CONFIGS:
        detail_path = output_dir / f"details_{chunk_size}_{overlap}.csv"
        meta_path = output_dir / f"meta_{chunk_size}_{overlap}.json"

        if not detail_path.exists() or not meta_path.exists():
            continue

        details = pd.read_csv(detail_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if len(details) != len(dataset):
            # Không trộn smoke-test với full-run.
            continue

        row = summarize_config(
            details=details,
            chunk_count=int(meta["total_chunks"]),
            n_source_rows=int(meta["source_rows_indexed"]),
            build_seconds=float(meta["build_seconds"]),
        )
        rows.append(row)

        topic_summary = (
            details.groupby("topic", as_index=False)
            .agg(
                n=("question_id", "count"),
                hit_at_1=("hit_at_1", "mean"),
                hit_at_3=("hit_at_3", "mean"),
                hit_at_5=("hit_at_5", "mean"),
                mrr_at_5=("rr_at_5", "mean"),
                median_latency_seconds=("latency_seconds", "median"),
            )
        )
        topic_summary.insert(0, "chunk_overlap", int(overlap))
        topic_summary.insert(0, "chunk_size", int(chunk_size))
        topic_frames.append(topic_summary)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            ["chunk_size", "chunk_overlap"]
        ).reset_index(drop=True)

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if topic_frames:
        pd.concat(topic_frames, ignore_index=True).to_csv(
            output_dir / "summary_by_topic.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = full 2213; ví dụ --limit 50 để smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        default="chunking_ablation_official_results",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help='Ví dụ: --configs "1800,360"',
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Tách smoke-test và full để không ghi đè nhau.
    if args.limit and args.limit > 0:
        output_dir = output_dir / f"smoke_{args.limit}"

    output_dir.mkdir(parents=True, exist_ok=True)

    configs = CHUNK_CONFIGS
    if args.configs:
        configs = []
        for item in args.configs:
            size, overlap = item.split(",", 1)
            configs.append((int(size), int(overlap)))

    print("=" * 96)
    print("MOMCARE CHUNKING ABLATION - OFFICIAL ViMedAQA CLEAN TEST")
    print("=" * 96)
    print(f"Dataset : {DATASET_NAME}")
    print(f"Config  : {DATASET_CONFIG}")
    print(f"Split   : {DATASET_SPLIT}")
    print(f"Expected: {EXPECTED_TEST_SAMPLES} official -> {EXPECTED_VALID_SAMPLES} clean")
    print(f"Configs : {configs}")
    print("Index   : CONTEXT ONLY")
    print("LLM     : KHÔNG gọi")
    print("Prod DB : KHÔNG bị thay đổi")
    print()

    dataset = load_official_clean_test(
        output_dir=output_dir,
        limit=args.limit,
        seed=args.seed,
    )

    embeddings = get_embeddings()

    for chunk_size, overlap in configs:
        print("\n" + "=" * 96)
        print(
            f"🧪 chunk_size={chunk_size}, overlap={overlap}"
        )
        print("=" * 96)

        started = time.perf_counter()

        docs = build_documents(
            dataset=dataset,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        lengths = [len(doc.page_content) for doc in docs]
        violations = sum(
            1 for length in lengths
            if length > chunk_size
        )

        print(f"  Source rows : {len(dataset):,}")
        print(f"  Total chunks: {len(docs):,}")
        print(
            f"  Chunk chars : "
            f"min={min(lengths) if lengths else 0}, "
            f"avg={np.mean(lengths) if lengths else 0:.1f}, "
            f"max={max(lengths) if lengths else 0}"
        )
        print(f"  Violations  : {violations}")

        if violations:
            raise RuntimeError(
                f"Có {violations} chunks vượt chunk_size={chunk_size}."
            )

        index_dir = (
            output_dir
            / f"experimental_index_{chunk_size}_{overlap}"
        )

        db, bm25, bm25_docs = build_faiss_and_bm25(
            docs=docs,
            embeddings=embeddings,
            index_dir=index_dir,
        )

        build_seconds = time.perf_counter() - started

        meta = {
            "dataset_name": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "n_queries": len(dataset),
            "source_rows_indexed": len(dataset),
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "total_chunks": len(docs),
            "min_chunk_chars": min(lengths) if lengths else 0,
            "avg_chunk_chars": float(np.mean(lengths)) if lengths else 0.0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "hard_limit_violations": violations,
            "build_seconds": build_seconds,
        }

        (
            output_dir
            / f"meta_{chunk_size}_{overlap}.json"
        ).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"  ✅ Build xong: {build_seconds:.1f}s")

        detail_path = (
            output_dir
            / f"details_{chunk_size}_{overlap}.csv"
        )

        details = evaluate_config(
            dataset=dataset,
            db=db,
            bm25=bm25,
            bm25_docs=bm25_docs,
            chunk_size=chunk_size,
            overlap=overlap,
            detail_path=detail_path,
        )

        row = summarize_config(
            details=details,
            chunk_count=len(docs),
            n_source_rows=len(dataset),
            build_seconds=build_seconds,
        )

        print(
            f"\n  📊 {chunk_size}/{overlap} | "
            f"Hit@1={row['hit_at_1']:.4f} | "
            f"Hit@3={row['hit_at_3']:.4f} | "
            f"Hit@5={row['hit_at_5']:.4f} | "
            f"MRR@5={row['mrr_at_5']:.4f}"
        )

        del details
        del db
        del bm25
        del bm25_docs
        del docs
        gc.collect()

        # Cập nhật summary sau từng config.
        rebuild_summary_from_details(
            output_dir=output_dir,
            dataset=dataset,
        )

    summary = rebuild_summary_from_details(
        output_dir=output_dir,
        dataset=dataset,
    )

    print("\n" + "=" * 110)
    print("TÓM TẮT CHUNKING ABLATION - OFFICIAL ViMedAQA CLEAN")
    print("=" * 110)

    if summary.empty:
        print("Chưa có đủ kết quả.")
        return

    display = summary.copy()

    for col in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5"]:
        display[col] = display[col].round(4)

    display["avg_chunks_per_source_row"] = (
        display["avg_chunks_per_source_row"].round(2)
    )
    display["mean_latency_seconds"] = (
        display["mean_latency_seconds"].round(4)
    )
    display["median_latency_seconds"] = (
        display["median_latency_seconds"].round(4)
    )
    display["build_seconds"] = (
        display["build_seconds"].round(1)
    )

    print(display.to_string(index=False))

    best = summary.sort_values(
        ["mrr_at_5", "hit_at_5", "hit_at_3"],
        ascending=False,
    ).iloc[0]

    print(
        "\n🏆 Cao nhất theo MRR@5 -> Hit@5 -> Hit@3:"
    )
    print(
        f"   {int(best['chunk_size'])}/"
        f"{int(best['chunk_overlap'])} | "
        f"Hit@5={best['hit_at_5']:.4f} | "
        f"MRR@5={best['mrr_at_5']:.4f}"
    )

    print(f"\n✅ Kết quả: {output_dir}")
    print("✅ Production VectorDB không bị thay đổi.")


if __name__ == "__main__":
    main()
