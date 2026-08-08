"""Leakage-controlled ViMedAQA benchmark for MomCare.

Stage 1 (build):
    - Load the official ViMedAQA `all/test` split (2,217 samples).
    - Build a dedicated FAISS index from CONTEXT ONLY.
    - Keep question/answer in a separate evaluation CSV.

Stage 2 (retrieve):
    - Query the context-only index with all 2,217 test questions.
    - Run the same dense + BM25 reciprocal-rank hybrid idea used by MomCare.
    - Report Hit@1, Hit@3, Hit@5, MRR@5 and latency.

This file never modifies MomCare's production VectorDB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover - only required for `generate`
    Groq = None


DATASET_NAME = "tmnam20/ViMedAQA"
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
EXPECTED_TEST_SAMPLES = 2217

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# Same final chunk constraint as MomCare.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 360

# Same final retrieval scale as MomCare.
DENSE_POOL_K = 50
BM25_POOL_K = 50
CANDIDATE_K = 25
FINAL_K = 5

# Same final generation constraints as MomCare.
MODEL_NAME = "llama-3.1-8b-instant"
GENERATION_MAX_TOKENS = 350
CONTEXT_MAX_TOKENS = 2200
VIETNAMESE_CHARS_PER_TOKEN = 2.3

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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*",
        normalize_text(text).lower(),
    )


def context_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def classify_query(question: str) -> str:
    """Mirror MomCare's final Adaptive Alpha query profiles."""
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
    """Keep MomCare's retrieval-only alias for complementary feeding."""
    original = normalize_text(question)
    lowered = original.lower()
    additions: list[str] = []

    if "ăn dặm" in lowered and "ăn bổ sung" not in lowered:
        additions.append("ăn bổ sung")

    if "ăn dặm" in lowered and any(
        marker in lowered for marker in ("có nên", "bắt đầu", "khi nào", "từ mấy tháng")
    ):
        additions.append("thời điểm bắt đầu ăn bổ sung")

    if any(marker in lowered for marker in ("mấy bữa", "bao nhiêu bữa", "số bữa")):
        additions.append("tần suất ăn bổ sung")

    return normalize_text(" ".join([original, *additions]))


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_official_test() -> pd.DataFrame:
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
    )
    df = dataset.to_pandas().reset_index(drop=True)

    required = {"question_idx", "question", "answer", "context", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"ViMedAQA thiếu các cột bắt buộc: {sorted(missing)}")
    if len(df) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"Sai số mẫu test: nhận {len(df)}, kỳ vọng {EXPECTED_TEST_SAMPLES}."
        )
    return df


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = out_dir / "faiss_context_only"
    chunks_path = out_dir / "context_only_chunks.jsonl"
    queries_path = out_dir / "test_queries_and_references.csv"
    excluded_path = out_dir / "excluded_samples.csv"

    df = load_official_test()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs: list[Document] = []
    eval_rows: list[dict] = []
    excluded_rows: list[dict] = []

    for row_index, row in df.iterrows():
        question = normalize_text(row["question"])
        answer = normalize_text(row["answer"])
        context = normalize_text(row["context"])
        missing_fields = [
            field_name
            for field_name, field_value in (
                ("question", question),
                ("answer", answer),
                ("context", context),
            )
            if not field_value
        ]

        # A clean retrieval+generation benchmark needs all three fields.
        # Do not fabricate missing context from the reference answer.
        if missing_fields:
            excluded_rows.append(
                {
                    "row_index": int(row_index),
                    "sample_id": str(row["question_idx"]),
                    "topic": str(row["topic"]),
                    "missing_fields": "|".join(missing_fields),
                    "reason": "missing_required_field_in_official_test_split",
                }
            )
            continue

        gold_hash = context_hash(context)
        topic_raw = row["topic"]
        try:
            topic = TOPIC_MAP.get(int(topic_raw), str(topic_raw))
        except (TypeError, ValueError):
            topic = str(topic_raw)

        # IMPORTANT: only `context` is passed into the splitter and FAISS.
        chunks = splitter.split_text(context)
        if not chunks:
            chunks = [context]

        for chunk_index, chunk in enumerate(chunks):
            chunk = normalize_text(chunk)
            if not chunk:
                continue
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "benchmark": "ViMedAQA",
                        "split": "test",
                        "row_index": int(row_index),
                        "context_hash": gold_hash,
                        "chunk_in_context": int(chunk_index),
                        "topic": topic,
                        "content_field": "context_only",
                    },
                )
            )

        # question/answer stay in this separate evaluation file only.
        eval_rows.append(
            {
                "row_index": int(row_index),
                "sample_id": str(row["question_idx"]),
                "topic": topic,
                "question": question,
                "reference_answer": answer,
                "gold_context_hash": gold_hash,
            }
        )

    if not docs:
        raise RuntimeError("Không tạo được context chunk nào.")

    max_chars = max(len(doc.page_content) for doc in docs)
    violations = sum(len(doc.page_content) > CHUNK_SIZE for doc in docs)
    forbidden_metadata = {"question", "answer", "reference_answer"}
    metadata_violations = sum(
        bool(forbidden_metadata.intersection(doc.metadata.keys())) for doc in docs
    )

    if violations or metadata_violations:
        raise RuntimeError(
            "Leakage/chunk audit thất bại: "
            f"chunk_violations={violations}, metadata_violations={metadata_violations}."
        )

    print("\n========== VIMEDAQA CLEAN INDEX AUDIT ==========")
    print(f"Official test samples : {len(df)}")
    print(f"Valid benchmark rows  : {len(eval_rows)}")
    print(f"Excluded rows         : {len(excluded_rows)}")
    print(f"Context-only chunks   : {len(docs)}")
    print(f"Max chunk chars       : {max_chars}/{CHUNK_SIZE}")
    print(f"Overlap               : {CHUNK_OVERLAP}")
    print(f"Hard-limit violations : {violations}")
    print(f"Q/A metadata leakage  : {metadata_violations}")
    print("Indexed content field : context ONLY")
    print("================================================\n")

    with chunks_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(
                json.dumps(
                    {"text": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )

    pd.DataFrame(eval_rows).to_csv(queries_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded_rows).to_csv(
        excluded_path,
        index=False,
        encoding="utf-8-sig",
    )

    db = FAISS.from_documents(docs, get_embeddings())
    db.save_local(str(index_dir))

    print(f"✅ Đã lưu FAISS benchmark riêng: {index_dir}")
    print(f"✅ Context chunks             : {chunks_path}")
    print(f"✅ Query/reference riêng      : {queries_path}")
    print(f"✅ Mẫu bị loại + lý do        : {excluded_path}")
    print("✅ Không thay đổi VectorDB production của MomCare.")


def load_context_docs(chunks_path: Path) -> list[Document]:
    docs: list[Document] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
    return docs


def doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    return (
        f"{md.get('context_hash', '')}|{md.get('chunk_in_context', '')}|"
        f"{normalize_text(doc.page_content)[:1000]}"
    )


def hybrid_search(
    question: str,
    db: FAISS,
    bm25: BM25Okapi,
    docs: list[Document],
) -> list[Document]:
    retrieval_question = expand_retrieval_query(question)
    profile = classify_query(question)
    alpha = ALPHA_BY_PROFILE[profile]

    dense_docs = db.similarity_search(retrieval_question, k=DENSE_POOL_K)
    bm25_scores = bm25.get_scores(tokenize(retrieval_question))
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: float(bm25_scores[i]),
        reverse=True,
    )[:BM25_POOL_K]
    bm25_docs = [docs[i] for i in bm25_indices]

    dense_rank = {doc_key(doc): rank for rank, doc in enumerate(dense_docs, 1)}
    lexical_rank = {doc_key(doc): rank for rank, doc in enumerate(bm25_docs, 1)}

    candidates: dict[str, Document] = {}
    for doc in dense_docs + bm25_docs:
        candidates.setdefault(doc_key(doc), doc)

    scored: list[tuple[float, Document]] = []
    for key, doc in candidates.items():
        vector_score = 1.0 / dense_rank[key] if key in dense_rank else 0.0
        lexical_score = 1.0 / lexical_rank[key] if key in lexical_rank else 0.0
        score = alpha * vector_score + (1.0 - alpha) * lexical_score
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:CANDIDATE_K]][:FINAL_K]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / VIETNAMESE_CHARS_PER_TOKEN) + 1)


def build_generation_context(top_docs: list[Document]) -> tuple[str, list[Document], int]:
    blocks: list[str] = []
    used_docs: list[Document] = []
    used_tokens = 0

    for doc in top_docs:
        content = normalize_text(doc.page_content)
        if not content:
            continue
        block = f'<TAI_LIEU id="{len(used_docs) + 1}">\n{content}\n</TAI_LIEU>'
        block_tokens = estimate_tokens(block)
        if used_tokens + block_tokens > CONTEXT_MAX_TOKENS:
            break
        blocks.append(block)
        used_docs.append(doc)
        used_tokens += block_tokens

    return "\n\n".join(blocks), used_docs, used_tokens


def load_groq_keys() -> list[str]:
    if load_dotenv is not None:
        load_dotenv()

    names = ("GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3")
    keys = [os.getenv(name, "").strip() for name in names]
    keys = [key for key in keys if key]
    if not keys:
        raise RuntimeError(
            "Không tìm thấy GROQ_API_KEY. Hãy dùng cùng file .env của MomCare."
        )
    return keys


def parse_retry_after_seconds(message: str) -> float | None:
    """Parse Groq messages such as `try again in 3m58.8096s`."""
    match = re.search(
        r"try again in\s+"
        r"(?:(?P<hours>\d+)h)?"
        r"(?:(?P<minutes>\d+)m)?"
        r"(?P<seconds>[\d.]+)s",
        str(message),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return hours * 3600.0 + minutes * 60.0 + seconds


class GroqKeyPool:
    """Round-robin Groq keys with per-key cooldown after a 429 response."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.cooldown_until = [0.0 for _ in self.keys]
        self.cursor = 0

    def acquire(self) -> tuple[int, str]:
        while True:
            now = time.monotonic()
            for offset in range(len(self.keys)):
                index = (self.cursor + offset) % len(self.keys)
                if self.cooldown_until[index] <= now:
                    self.cursor = (index + 1) % len(self.keys)
                    return index, self.keys[index]

            wait_s = max(0.5, min(self.cooldown_until) - now + 0.5)
            print(
                f"⏳ [GROQ TPD] Tất cả API key đang cooldown; "
                f"chờ {wait_s:.1f}s..."
            )
            time.sleep(wait_s)

    def cooldown(self, index: int, seconds: float) -> None:
        seconds = max(1.0, float(seconds))
        self.cooldown_until[index] = max(
            self.cooldown_until[index],
            time.monotonic() + seconds + 0.5,
        )


def call_benchmark_llm(
    question: str,
    context: str,
    key_pool: GroqKeyPool,
) -> tuple[str, int, int]:
    if Groq is None:
        raise RuntimeError("Thiếu package `groq`. Cài bằng: pip install groq")

    system_prompt = (
        "Bạn là hệ thống hỏi đáp y tế tiếng Việt trong một thí nghiệm benchmark. "
        "Chỉ sử dụng thông tin trong NGỮ CẢNH được cung cấp. "
        "Không bổ sung kiến thức bên ngoài. "
        "Trả lời trực tiếp, ngắn gọn, giữ nguyên số liệu và đơn vị. "
        "Nếu ngữ cảnh không đủ căn cứ, hãy nói rõ chưa đủ thông tin."
    )
    user_prompt = f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}\n\nTRẢ LỜI:"

    last_error: Exception | None = None
    non_rate_errors = 0

    while True:
        key_index, api_key = key_pool.acquire()
        try:
            client = Groq(api_key=api_key, timeout=30.0, max_retries=0)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=GENERATION_MAX_TOKENS,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            answer = normalize_text(response.choices[0].message.content)
            return answer, prompt_tokens, completion_tokens
        except Exception as error:  # network/rate-limit path
            last_error = error
            message = str(error)
            if "429" in message or "rate_limit" in message.lower():
                retry_s = parse_retry_after_seconds(message)
                if retry_s is None:
                    retry_s = 60.0
                key_pool.cooldown(key_index, retry_s)
                print(
                    f"⏳ [GROQ RATE LIMIT] key#{key_index + 1} "
                    f"cooldown {retry_s:.1f}s"
                )
                # A 429 is not a failed benchmark sample. Try another key;
                # if all keys are cooling down, acquire() waits automatically.
                continue
            else:
                non_rate_errors += 1
                if non_rate_errors >= 4:
                    raise RuntimeError(f"Groq thất bại sau retry: {last_error}")
                time.sleep(min(8.0, float(non_rate_errors)))


def existing_completed_ids(checkpoint_path: Path) -> set[int]:
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return set()
    completed: set[int] = set()
    with checkpoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).strip() == "OK":
                completed.add(int(row["row_index"]))
    return completed


def generate(out_dir: Path, limit: int | None = None) -> None:
    """Generate benchmark answers from the dedicated context-only index.

    This deliberately evaluates the RAG core, not MomCare's product-level
    OUT_OF_SCOPE/Input Guardrails, because ViMedAQA covers general medicine.
    """
    index_dir = out_dir / "faiss_context_only"
    chunks_path = out_dir / "context_only_chunks.jsonl"
    queries_path = out_dir / "test_queries_and_references.csv"
    checkpoint_path = out_dir / "generation_results.csv"

    if not index_dir.exists() or not chunks_path.exists() or not queries_path.exists():
        raise RuntimeError("Chưa có clean benchmark index. Chạy `build` trước.")

    docs = load_context_docs(chunks_path)
    queries = pd.read_csv(queries_path)
    db = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])
    key_pool = GroqKeyPool(load_groq_keys())
    completed = existing_completed_ids(checkpoint_path)

    fieldnames = [
        "row_index", "sample_id", "topic", "question", "reference_answer",
        "prediction", "gold_context_rank", "gold_context_hit5",
        "retrieved_context_hashes", "generation_context_hashes",
        "context_estimated_tokens", "prompt_tokens", "completion_tokens",
        "latency_s", "status", "error",
    ]
    new_file = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    processed_this_run = 0

    with checkpoint_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
            handle.flush()

        for _, row in queries.iterrows():
            row_index = int(row["row_index"])
            if row_index in completed:
                continue
            if limit is not None and processed_this_run >= limit:
                break

            start = time.perf_counter()
            top_docs: list[Document] = []
            try:
                question = str(row["question"])
                gold_hash = str(row["gold_context_hash"])
                top_docs = hybrid_search(question, db, bm25, docs)

                gold_rank = None
                for rank, doc in enumerate(top_docs, 1):
                    if str(doc.metadata.get("context_hash")) == gold_hash:
                        gold_rank = rank
                        break

                context, used_docs, context_tokens = build_generation_context(top_docs)
                if not context:
                    raise RuntimeError("Không tạo được generation context trong budget.")

                prediction, prompt_tokens, completion_tokens = call_benchmark_llm(
                    question,
                    context,
                    key_pool,
                )
                status = "OK"
                error_text = ""
            except Exception as error:
                prediction = ""
                prompt_tokens = 0
                completion_tokens = 0
                context_tokens = 0
                used_docs = []
                gold_rank = None
                status = "ERROR"
                error_text = str(error)[:500]

            latency = time.perf_counter() - start
            writer.writerow(
                {
                    "row_index": row_index,
                    "sample_id": row["sample_id"],
                    "topic": row["topic"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "prediction": prediction,
                    "gold_context_rank": gold_rank if gold_rank is not None else "",
                    "gold_context_hit5": gold_rank is not None and gold_rank <= 5,
                    "retrieved_context_hashes": "|".join(
                        str(doc.metadata.get("context_hash", "")) for doc in top_docs
                    ),
                    "generation_context_hashes": "|".join(
                        str(doc.metadata.get("context_hash", "")) for doc in used_docs
                    ),
                    "context_estimated_tokens": context_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_s": latency,
                    "status": status,
                    "error": error_text,
                }
            )
            handle.flush()
            processed_this_run += 1

            if processed_this_run % 10 == 0 or status != "OK":
                print(
                    f"Generation this run: {processed_this_run} | "
                    f"row={row_index} | status={status} | "
                    f"gold_rank={gold_rank if gold_rank is not None else '-'} | "
                    f"time={latency:.2f}s"
                )

    all_rows = []
    with checkpoint_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    completed_ok_ids = {
        int(row["row_index"])
        for row in all_rows
        if row.get("status") == "OK"
    }
    ok_count = len(completed_ok_ids)
    historical_error_count = sum(row.get("status") == "ERROR" for row in all_rows)
    unresolved_count = len(queries) - ok_count

    print("\n========== VIMEDAQA CLEAN GENERATION ==========")
    print(f"Completed OK      : {ok_count}")
    print(f"Unresolved        : {unresolved_count}")
    print(f"Historical errors : {historical_error_count}")
    print(f"Target       : {len(queries)}")
    print(f"Checkpoint   : {checkpoint_path}")
    print("===============================================\n")


def retrieve(out_dir: Path) -> None:
    index_dir = out_dir / "faiss_context_only"
    chunks_path = out_dir / "context_only_chunks.jsonl"
    queries_path = out_dir / "test_queries_and_references.csv"
    excluded_path = out_dir / "excluded_samples.csv"
    output_path = out_dir / "retrieval_results.csv"

    if not index_dir.exists() or not chunks_path.exists() or not queries_path.exists():
        raise RuntimeError(
            "Chưa có clean benchmark index. Chạy lệnh `build` trước."
        )

    docs = load_context_docs(chunks_path)
    queries = pd.read_csv(queries_path)
    excluded_count = 0
    if excluded_path.exists():
        excluded_count = len(pd.read_csv(excluded_path))

    if len(queries) + excluded_count != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Số mẫu benchmark không khớp official test split: "
            f"valid={len(queries)}, excluded={excluded_count}, "
            f"expected={EXPECTED_TEST_SAMPLES}."
        )

    db = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])

    result_rows: list[dict] = []
    reciprocal_ranks: list[float] = []
    hit1 = hit3 = hit5 = 0
    latencies: list[float] = []

    for idx, row in queries.iterrows():
        question = str(row["question"])
        gold_hash = str(row["gold_context_hash"])

        start = time.perf_counter()
        top_docs = hybrid_search(question, db, bm25, docs)
        latency = time.perf_counter() - start
        latencies.append(latency)

        first_rank = None
        for rank, doc in enumerate(top_docs, 1):
            if str(doc.metadata.get("context_hash")) == gold_hash:
                first_rank = rank
                break

        if first_rank == 1:
            hit1 += 1
        if first_rank is not None and first_rank <= 3:
            hit3 += 1
        if first_rank is not None and first_rank <= 5:
            hit5 += 1

        rr = 1.0 / first_rank if first_rank is not None and first_rank <= 5 else 0.0
        reciprocal_ranks.append(rr)

        result_rows.append(
            {
                "row_index": int(row["row_index"]),
                "sample_id": row["sample_id"],
                "topic": row["topic"],
                "question": question,
                "first_gold_rank": first_rank if first_rank is not None else "",
                "hit_at_1": first_rank == 1,
                "hit_at_3": first_rank is not None and first_rank <= 3,
                "hit_at_5": first_rank is not None and first_rank <= 5,
                "latency_s": latency,
                "retrieved_context_hashes": "|".join(
                    str(doc.metadata.get("context_hash", "")) for doc in top_docs
                ),
            }
        )

        if (idx + 1) % 100 == 0 or idx + 1 == len(queries):
            print(f"Retrieval: {idx + 1}/{len(queries)}")

    pd.DataFrame(result_rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    n = len(queries)
    mrr5 = sum(reciprocal_ranks) / n
    avg_latency = sum(latencies) / len(latencies)

    print("\n========== VIMEDAQA CLEAN RETRIEVAL ==========")
    print(f"Queries : {n}")
    print(f"Excluded: {excluded_count}")
    print(f"Hit@1   : {100.0 * hit1 / n:.2f}%")
    print(f"Hit@3   : {100.0 * hit3 / n:.2f}%")
    print(f"Hit@5   : {100.0 * hit5 / n:.2f}%")
    print(f"MRR@5   : {mrr5:.4f}")
    print(f"Latency : {avg_latency:.4f}s/query")
    print(f"CSV     : {output_path}")
    print("=============================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "retrieve", "generate"))
    parser.add_argument(
        "--out-dir",
        default="evaluation/vimedaqa_clean",
        help="Dedicated benchmark output directory; production DB is never touched.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For `generate`: process only N new samples, useful for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.stage == "build":
        build(out_dir)
    elif args.stage == "retrieve":
        retrieve(out_dir)
    else:
        generate(out_dir, limit=args.limit)


if __name__ == "__main__":
    main()
