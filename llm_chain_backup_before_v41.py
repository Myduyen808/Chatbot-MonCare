#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_chunking_ablation.py
==============================

Chunking ablation an toàn cho MomCare, KHÔNG đụng vào VectorDB production.

Mục tiêu:
- So sánh 4 cấu hình chunk_size / overlap:
    512 / 100
    1000 / 200
    1800 / 360   <- production
    3000 / 600
- Chỉ lập chỉ mục trường `context` của ViMedAQA clean.
- `question` chỉ dùng làm truy vấn.
- `answer` không đưa vào index.
- Retrieval: FAISS + BM25 + Adaptive Alpha, Top-K = 5.
- Không gọi Groq/LLM.

Lưu ý quan trọng:
- KHÔNG dùng create_vectordb_with_file() của vectordb.py vì code production
  có hard cap MAX_CHUNK_CHARS = 1800; nếu dùng trực tiếp thì cấu hình
  3000/600 sẽ bị clamp về 1800/360 và ablation sẽ sai.
- Script này xây các index THỬ NGHIỆM riêng trong chunking_ablation_results/.
- VectorDB production hiện tại không bị xóa, đổi tên hay ghi đè.

Chạy thử:
    python benchmark_chunking_ablation.py --limit 50

Chạy toàn bộ:
    python benchmark_chunking_ablation.py

Nếu auto-detect không tìm thấy file ViMedAQA:
    python benchmark_chunking_ablation.py --input "duong_dan_file.xlsx"

Kết quả:
    chunking_ablation_results/
        sampled_or_full_dataset.csv
        details_512_100.csv
        details_1000_200.csv
        details_1800_360.csv
        details_3000_600.csv
        summary.csv
        summary_by_topic.csv
        experimental_index_512_100/
        ...
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CẤU HÌNH
# ============================================================

TOP_K = 5
DENSE_POOL_K = 50
BM25_POOL_K = 50
RANDOM_SEED = 42

CHUNK_CONFIGS = [
    (512, 100),
    (1000, 200),
    (1800, 360),
    (3000, 600),
]

TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*"
)

DEFAULT_ALPHA = {
    "exact_lexical": 0.20,
    "noisy_conversational": 0.30,
    "quantitative": 0.40,
    "semantic": 0.30,
    "table_bonus": 0.15,
}

TOPIC_MAP = {
    0: "Body Part",
    1: "Disease",
    2: "Drug",
    3: "Medicine",
}


# ============================================================
# TIỆN ÍCH
# ============================================================

def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_lower(value: object) -> str:
    return normalize_text(value).lower()


def stable_context_id(context: str) -> str:
    normalized = normalize_lower(context)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or "").lower())


def is_data_driven_chunk(text: str) -> bool:
    """
    Bám gần logic production:
    >30% dòng có dữ liệu định lượng -> data_table.
    """
    lines = str(text or "").split("\n")
    if len(lines) <= 2:
        return False

    data_lines = 0
    pattern = re.compile(
        r"(\d+\s*(mg|ml|g|%|tháng|tuổi|ngày|lần|tuần|kcal|kg))",
        flags=re.IGNORECASE,
    )

    for line in lines:
        if pattern.search(line):
            data_lines += 1

    return (data_lines / len(lines)) > 0.30


def classify_query(question: str) -> str:
    """
    Bám logic production hiện tại trong llm_chain.py:
    quantitative -> exact_lexical -> noisy_conversational -> semantic.
    """
    q = normalize_lower(question)

    quantitative_markers = [
        "bao nhiêu",
        "mấy lần",
        "mấy bữa",
        "mấy ngày",
        "mấy tháng",
        "mấy tuần",
        "bao lâu",
        "mỗi ngày",
        "mỗi tuần",
        "mỗi lần",
        "liều",
        "liều lượng",
        "tần suất",
        "số lượng",
        "lượng bao nhiêu",
    ]
    if any(marker in q for marker in quantitative_markers):
        return "quantitative"

    measurement_pattern = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml|g|kg|%|iu|kcal)\b",
        flags=re.IGNORECASE,
    )
    if measurement_pattern.search(q):
        return "quantitative"

    exact_terms = [
        "vitamin",
        "vitamin d",
        "paracetamol",
        "ibuprofen",
        "amoxicillin",
        "oxytocin",
        "aspirin",
        "sắt",
        "canxi",
        "axit folic",
        "tắc tia sữa",
        "viêm tuyến vú",
        "băng huyết",
        "sản dịch",
        "vàng da",
        "tưa miệng",
        "ăn dặm",
        "bú mẹ",
        "sữa mẹ",
    ]
    if any(term in q for term in exact_terms):
        return "exact_lexical"

    noisy_markers = [
        "mom",
        "mẹ ơi",
        "bé nhà em",
        "bé nhà mình",
        "ạ",
        "nha",
        "nhỉ",
        "kiểu",
        "sao á",
        "vậy ta",
        "hông",
        "hong",
        "ko ",
        "k ",
        "mik",
        "mn",
        "rồi á",
    ]
    if any(marker in q for marker in noisy_markers):
        return "noisy_conversational"

    return "semantic"


def expand_retrieval_query(question: str) -> str:
    """
    Bám logic alias production cho ăn dặm / ăn bổ sung.
    """
    original = normalize_text(question)
    lowered = original.lower()
    additions: list[str] = []

    group = ("ăn dặm", "ăn bổ sung")
    if any(term in lowered for term in group):
        for term in group:
            if term not in lowered and term not in additions:
                additions.append(term)

    feeding_time_patterns = (
        "có nên",
        "bắt đầu",
        "khi nào",
        "từ mấy tháng",
    )
    if (
        "ăn dặm" in lowered
        and any(p in lowered for p in feeding_time_patterns)
    ):
        phrase = "thời điểm bắt đầu ăn bổ sung"
        if phrase not in additions:
            additions.append(phrase)

    feeding_frequency_patterns = (
        "mấy bữa",
        "bao nhiêu bữa",
        "số bữa",
    )
    if any(p in lowered for p in feeding_frequency_patterns):
        phrase = "tần suất ăn bổ sung"
        if phrase not in additions:
            additions.append(phrase)

    if additions:
        return original + " " + " ".join(additions)

    return original


def load_alpha_config() -> dict[str, float]:
    config = dict(DEFAULT_ALPHA)
    path = Path("adaptive_alpha_config.json")

    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            for key in config:
                if key in loaded:
                    config[key] = float(loaded[key])
            print(f"✅ Adaptive Alpha config: {config}")
        except Exception as exc:
            print(f"⚠️ Không đọc được {path}: {exc}")
            print(f"   Dùng fallback: {config}")
    else:
        print(f"⚠️ Không thấy adaptive_alpha_config.json")
        print(f"   Dùng fallback: {config}")

    return config


# ============================================================
# TÌM VÀ ĐỌC ViMedAQA
# ============================================================

def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Không hỗ trợ định dạng: {path}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    if "question" not in mapping or "context" not in mapping:
        raise ValueError(
            "Dataset phải có ít nhất 2 cột: question và context."
        )

    out = pd.DataFrame()
    out["question"] = df[mapping["question"]]
    out["context"] = df[mapping["context"]]

    if "answer" in mapping:
        out["answer"] = df[mapping["answer"]]
    else:
        out["answer"] = ""

    if "topic" in mapping:
        out["topic"] = df[mapping["topic"]]
    else:
        out["topic"] = ""

    if "title" in mapping:
        out["title"] = df[mapping["title"]]
    else:
        out["title"] = ""

    return out


def auto_find_vimedaqa() -> Path:
    """
    Tìm file có question + context, ưu tiên file khoảng 2217 dòng
    và tên có vimedaqa/test.
    """
    roots = [
        Path("."),
        Path("data_store"),
        Path("data"),
        Path("dataset"),
        Path("datasets"),
        Path("benchmark"),
        Path("benchmarks"),
    ]

    candidates: list[tuple[float, Path, int]] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue

        for pattern in ("*.xlsx", "*.xls", "*.csv", "*.json", "*.jsonl"):
            for path in root.rglob(pattern):
                try:
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)

                    # Bỏ file output benchmark của chính script.
                    if "chunking_ablation_results" in str(path):
                        continue

                    df = read_tabular(path)
                    cols = {str(c).strip().lower() for c in df.columns}
                    if not {"question", "context"}.issubset(cols):
                        continue

                    n = len(df)
                    name = path.name.lower()

                    score = 0.0
                    if "vimedaqa" in name:
                        score += 100.0
                    if "test" in name:
                        score += 50.0

                    # gần 2217 dòng thì ưu tiên mạnh
                    score += max(0.0, 30.0 - abs(n - 2217) / 50.0)

                    # file quá nhỏ không phù hợp benchmark chính
                    if n < 100:
                        score -= 50.0

                    candidates.append((score, path, n))

                except Exception:
                    continue

    if not candidates:
        raise FileNotFoundError(
            "Không tự tìm thấy dataset có cột question + context. "
            "Hãy chạy lại với --input \"duong_dan_file.xlsx\"."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)

    print("\n🔎 Dataset ứng viên:")
    for score, path, n in candidates[:5]:
        print(f"  score={score:6.1f} | rows={n:5d} | {path}")

    selected = candidates[0][1]
    print(f"\n✅ Auto-select: {selected}\n")
    return selected


def load_clean_dataset(
    input_path: Path,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    raw = read_tabular(input_path)
    df = standardize_columns(raw)

    for col in ["question", "context", "answer", "title"]:
        df[col] = df[col].map(normalize_text)

    # topic giữ nguyên trước khi map
    df["topic_raw"] = df["topic"]

    # Empty context / question bị loại.
    df = df[
        (df["question"].str.len() > 0)
        & (df["context"].str.len() > 0)
        & (df["context"].str.lower() != "nan")
    ].copy()

    # Loại context dạng NaN sau normalize.
    df = df[df["context"].str.lower().ne("none")].copy()

    # Gold id dựa trên chính context, để duplicate context không bị tính miss.
    df["gold_context_id"] = df["context"].map(stable_context_id)

    def topic_name(value: object) -> str:
        if pd.isna(value):
            return "Unknown"

        try:
            number = int(float(value))
            return TOPIC_MAP.get(number, str(number))
        except Exception:
            text = str(value).strip()
            return text if text else "Unknown"

    df["topic"] = df["topic_raw"].map(topic_name)
    df["query_profile"] = df["question"].map(classify_query)

    df = df.reset_index(drop=False).rename(columns={"index": "original_row"})
    df["original_row"] = df["original_row"] + 2  # gần với số dòng Excel

    if limit and limit > 0 and limit < len(df):
        # Sample cân bằng theo topic nếu có thể.
        rng_parts = []
        groups = list(df.groupby("topic"))
        per_group = max(1, limit // max(1, len(groups)))

        for _, group in groups:
            n = min(per_group, len(group))
            rng_parts.append(group.sample(n=n, random_state=seed))

        sampled = pd.concat(rng_parts, ignore_index=False)

        if len(sampled) < limit:
            remaining = df.drop(index=sampled.index, errors="ignore")
            extra_n = min(limit - len(sampled), len(remaining))
            if extra_n > 0:
                sampled = pd.concat([
                    sampled,
                    remaining.sample(n=extra_n, random_state=seed),
                ])

        df = sampled.head(limit).sample(frac=1, random_state=seed).reset_index(drop=True)

    df.insert(
        0,
        "question_id",
        [f"Q{i:04d}" for i in range(1, len(df) + 1)],
    )

    return df


# ============================================================
# EMBEDDING
# ============================================================

def load_embedding_model() -> HuggingFaceEmbeddings:
    model_path = None

    config_path = Path("model_config.yml")
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model_path = config.get("embedding_path")
        except Exception as exc:
            print(f"⚠️ Không đọc được model_config.yml: {exc}")

    if not model_path:
        raise RuntimeError(
            "Không tìm thấy embedding_path trong model_config.yml. "
            "Script cần dùng đúng Vietnamese-SBERT của project."
        )

    print(f"🧠 Embedding model: {model_path}")
    return HuggingFaceEmbeddings(model_name=model_path)


# ============================================================
# CHUNKING VÀ INDEX
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

    # Chỉ index unique context để tránh duplicate documents.
    unique_contexts = (
        dataset[
            ["gold_context_id", "context", "topic", "title"]
        ]
        .drop_duplicates(subset=["gold_context_id"])
        .reset_index(drop=True)
    )

    chunks: list[Document] = []

    for _, row in unique_contexts.iterrows():
        context = row["context"]

        base_doc = Document(
            page_content=context,
            metadata={
                "gold_context_id": row["gold_context_id"],
                "topic": row["topic"],
                "title": row["title"],
                "source": "ViMedAQA_clean",
            },
        )

        split_docs = splitter.split_documents([base_doc])

        for local_idx, doc in enumerate(split_docs):
            text = normalize_text(doc.page_content)
            if not text:
                continue

            metadata = dict(doc.metadata or {})
            metadata["chunk_local_id"] = int(local_idx)
            metadata["chunk_type"] = (
                "data_table"
                if is_data_driven_chunk(doc.page_content)
                else "normal_text"
            )

            chunks.append(
                Document(
                    page_content=doc.page_content.strip(),
                    metadata=metadata,
                )
            )

    return chunks


@dataclass
class RetrievalBundle:
    db: FAISS
    bm25: BM25Okapi
    valid_docs: list[Document]


def build_retrieval_bundle(
    chunks: list[Document],
    embedding: HuggingFaceEmbeddings,
    index_dir: Path,
) -> RetrievalBundle:
    if not chunks:
        raise RuntimeError("Không có chunk để tạo index.")

    print(f"  🧱 Tạo FAISS từ {len(chunks):,} chunks...")
    db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding,
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(index_dir))

    print("  🔤 Tạo BM25...")
    valid_docs: list[Document] = []
    corpus: list[list[str]] = []

    for doc in chunks:
        tokens = tokenize(doc.page_content)
        if not tokens:
            continue
        valid_docs.append(doc)
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)

    return RetrievalBundle(
        db=db,
        bm25=bm25,
        valid_docs=valid_docs,
    )


# ============================================================
# ADAPTIVE HYBRID RETRIEVAL
# ============================================================

def doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    content = normalize_text(doc.page_content)[:1000]
    return (
        f"{md.get('gold_context_id','')}|"
        f"{md.get('chunk_local_id','')}|"
        f"{content}"
    )


def adaptive_hybrid_retrieve(
    question: str,
    bundle: RetrievalBundle,
    alpha_config: dict[str, float],
    top_k: int = TOP_K,
) -> tuple[list[Document], str, float]:
    profile = classify_query(question)
    retrieval_question = expand_retrieval_query(question)

    alpha = float(
        alpha_config.get(
            profile,
            DEFAULT_ALPHA.get(profile, 0.30),
        )
    )
    alpha = max(0.0, min(alpha, 1.0))

    table_bonus = max(
        0.0,
        float(alpha_config.get("table_bonus", 0.15)),
    )

    # Dense: 50 ứng viên.
    try:
        dense_docs = bundle.db.similarity_search(
            retrieval_question,
            k=DENSE_POOL_K,
            fetch_k=max(DENSE_POOL_K * 3, 150),
        )
    except TypeError:
        dense_docs = bundle.db.similarity_search(
            retrieval_question,
            k=DENSE_POOL_K,
        )

    # BM25: 50 ứng viên.
    q_tokens = tokenize(retrieval_question)
    scores = bundle.bm25.get_scores(q_tokens)

    top_indices = np.argsort(scores)[::-1][:BM25_POOL_K]
    bm25_docs = [
        bundle.valid_docs[int(idx)]
        for idx in top_indices
        if 0 <= int(idx) < len(bundle.valid_docs)
    ]

    dense_rank = {
        doc_key(doc): rank
        for rank, doc in enumerate(dense_docs, start=1)
    }

    bm25_rank = {
        doc_key(doc): rank
        for rank, doc in enumerate(bm25_docs, start=1)
    }

    candidates: dict[str, Document] = {}
    for doc in dense_docs + bm25_docs:
        candidates.setdefault(doc_key(doc), doc)

    ranked: list[tuple[float, Document]] = []

    for key, doc in candidates.items():
        dense_score = (
            1.0 / dense_rank[key]
            if key in dense_rank
            else 0.0
        )
        lexical_score = (
            1.0 / bm25_rank[key]
            if key in bm25_rank
            else 0.0
        )

        score = (
            alpha * dense_score
            + (1.0 - alpha) * lexical_score
        )

        if (
            profile == "quantitative"
            and (doc.metadata or {}).get("chunk_type") == "data_table"
        ):
            score += table_bonus

        ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return (
        [doc for _, doc in ranked[:top_k]],
        profile,
        alpha,
    )


# ============================================================
# METRICS
# ============================================================

def rank_of_gold(
    docs: list[Document],
    gold_context_id: str,
    max_k: int = 5,
) -> int | None:
    for rank, doc in enumerate(docs[:max_k], start=1):
        if (doc.metadata or {}).get("gold_context_id") == gold_context_id:
            return rank
    return None


def evaluate_config(
    dataset: pd.DataFrame,
    bundle: RetrievalBundle,
    chunk_size: int,
    overlap: int,
    alpha_config: dict[str, float],
) -> pd.DataFrame:
    rows = []
    total = len(dataset)

    for idx, row in dataset.iterrows():
        start = time.perf_counter()

        docs, profile, alpha = adaptive_hybrid_retrieve(
            question=row["question"],
            bundle=bundle,
            alpha_config=alpha_config,
            top_k=TOP_K,
        )

        latency = time.perf_counter() - start

        rank = rank_of_gold(
            docs,
            row["gold_context_id"],
            max_k=TOP_K,
        )

        retrieved_ids = [
            str((doc.metadata or {}).get("gold_context_id", ""))
            for doc in docs
        ]

        rows.append({
            "question_id": row["question_id"],
            "original_row": int(row["original_row"]),
            "topic": row["topic"],
            "query_profile": profile,
            "question": row["question"],
            "gold_context_id": row["gold_context_id"],
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(overlap),
            "selected_alpha": float(alpha),
            "gold_rank": rank if rank is not None else 0,
            "hit_at_1": int(rank == 1),
            "hit_at_3": int(rank is not None and rank <= 3),
            "hit_at_5": int(rank is not None and rank <= 5),
            "rr_at_5": (1.0 / rank) if rank is not None else 0.0,
            "latency_seconds": latency,
            "retrieved_context_ids": " | ".join(retrieved_ids),
        })

        current = len(rows)
        if current % 50 == 0 or current == total:
            print(
                f"    [{current:>4}/{total}] "
                f"Hit@5={np.mean([r['hit_at_5'] for r in rows]):.4f} | "
                f"MRR@5={np.mean([r['rr_at_5'] for r in rows]):.4f}"
            )

    return pd.DataFrame(rows)


def summarize(
    details: pd.DataFrame,
    chunk_count: int,
    unique_contexts: int,
) -> dict:
    return {
        "chunk_size": int(details["chunk_size"].iloc[0]),
        "chunk_overlap": int(details["chunk_overlap"].iloc[0]),
        "n_queries": int(len(details)),
        "unique_contexts": int(unique_contexts),
        "total_chunks": int(chunk_count),
        "avg_chunks_per_context": float(chunk_count / max(1, unique_contexts)),
        "hit_at_1": float(details["hit_at_1"].mean()),
        "hit_at_3": float(details["hit_at_3"].mean()),
        "hit_at_5": float(details["hit_at_5"].mean()),
        "mrr_at_5": float(details["rr_at_5"].mean()),
        "mean_latency_seconds": float(details["latency_seconds"].mean()),
        "median_latency_seconds": float(details["latency_seconds"].median()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Đường dẫn ViMedAQA test. Bỏ trống để auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = toàn bộ; ví dụ --limit 50 để smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chunking_ablation_results",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help='Tùy chọn, ví dụ: --configs "512,100" "1800,360"',
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
    else:
        input_path = auto_find_vimedaqa()

    configs = CHUNK_CONFIGS
    if args.configs:
        parsed = []
        for item in args.configs:
            size, overlap = item.split(",", 1)
            parsed.append((int(size), int(overlap)))
        configs = parsed

    print("=" * 88)
    print("CHUNKING ABLATION - MOMCARE / ViMedAQA CLEAN")
    print("=" * 88)
    print(f"Dataset: {input_path}")
    print(f"Configs: {configs}")
    print("Production VectorDB: KHÔNG BỊ THAY ĐỔI")
    print("Index: chỉ context; question/answer không đưa vào index.")
    print()

    dataset = load_clean_dataset(
        input_path=input_path,
        limit=args.limit,
        seed=args.seed,
    )

    dataset.to_csv(
        output_dir / "sampled_or_full_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✅ Số câu hợp lệ dùng đánh giá: {len(dataset):,}")
    print(
        f"✅ Unique gold contexts: "
        f"{dataset['gold_context_id'].nunique():,}"
    )
    print("📌 Topic:")
    print(dataset["topic"].value_counts(dropna=False).to_string())
    print()

    alpha_config = load_alpha_config()
    embedding = load_embedding_model()

    summary_rows = []
    topic_frames = []

    for chunk_size, overlap in configs:
        print("\n" + "=" * 88)
        print(
            f"🧪 CONFIG chunk_size={chunk_size}, "
            f"overlap={overlap}"
        )
        print("=" * 88)

        started_build = time.perf_counter()

        chunks = build_documents(
            dataset=dataset,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        lengths = [len(doc.page_content) for doc in chunks]

        print(f"  Tổng chunks: {len(chunks):,}")
        print(
            f"  Kích thước chunk: "
            f"min={min(lengths) if lengths else 0}, "
            f"avg={np.mean(lengths) if lengths else 0:.1f}, "
            f"max={max(lengths) if lengths else 0}"
        )

        index_dir = (
            output_dir
            / f"experimental_index_{chunk_size}_{overlap}"
        )

        bundle = build_retrieval_bundle(
            chunks=chunks,
            embedding=embedding,
            index_dir=index_dir,
        )

        build_seconds = time.perf_counter() - started_build

        print(
            f"  ✅ Build xong trong {build_seconds:.1f}s. "
            f"Bắt đầu đánh giá {len(dataset):,} queries..."
        )

        details = evaluate_config(
            dataset=dataset,
            bundle=bundle,
            chunk_size=chunk_size,
            overlap=overlap,
            alpha_config=alpha_config,
        )

        detail_path = (
            output_dir
            / f"details_{chunk_size}_{overlap}.csv"
        )
        details.to_csv(
            detail_path,
            index=False,
            encoding="utf-8-sig",
        )

        row = summarize(
            details=details,
            chunk_count=len(chunks),
            unique_contexts=dataset["gold_context_id"].nunique(),
        )
        row["build_seconds"] = float(build_seconds)
        summary_rows.append(row)

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

        # checkpoint summary sau mỗi cấu hình
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\n  📊 Kết quả {chunk_size}/{overlap}: "
            f"Hit@1={row['hit_at_1']:.4f} | "
            f"Hit@3={row['hit_at_3']:.4f} | "
            f"Hit@5={row['hit_at_5']:.4f} | "
            f"MRR@5={row['mrr_at_5']:.4f}"
        )

        # Giải phóng index/BM25 trước khi build cấu hình tiếp theo.
        del bundle
        del chunks
        del details
        gc.collect()

    summary = pd.DataFrame(summary_rows)

    if topic_frames:
        by_topic = pd.concat(topic_frames, ignore_index=True)
    else:
        by_topic = pd.DataFrame()

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_topic.to_csv(
        output_dir / "summary_by_topic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("TÓM TẮT CHUNKING ABLATION")
    print("=" * 100)

    display = summary.copy()
    for col in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5"]:
        display[col] = display[col].round(4)

    for col in ["mean_latency_seconds", "median_latency_seconds"]:
        display[col] = display[col].round(4)

    display["avg_chunks_per_context"] = (
        display["avg_chunks_per_context"].round(2)
    )
    display["build_seconds"] = display["build_seconds"].round(1)

    print(display.to_string(index=False))

    if not summary.empty:
        best = summary.sort_values(
            ["mrr_at_5", "hit_at_5", "hit_at_3"],
            ascending=False,
        ).iloc[0]

        print(
            "\n🏆 Tốt nhất theo thứ tự ưu tiên "
            "MRR@5 -> Hit@5 -> Hit@3:"
        )
        print(
            f"   chunk={int(best['chunk_size'])}/"
            f"{int(best['chunk_overlap'])} | "
            f"Hit@5={best['hit_at_5']:.4f} | "
            f"MRR@5={best['mrr_at_5']:.4f}"
        )

    print(f"\n✅ Chi tiết và summary nằm tại: {output_dir}")
    print(
        "✅ Production VectorDB không bị sửa. "
        "Có thể xóa các experimental_index_* sau khi lưu CSV."
    )


if __name__ == "__main__":
    main()
