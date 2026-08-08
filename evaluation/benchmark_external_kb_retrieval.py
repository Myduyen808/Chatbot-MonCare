import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

import vectordb
import llm_chain


DB_PATH = PROJECT_ROOT / "faiss_index_covid"
TEST_FILE = PROJECT_ROOT / "KB_COVID_VN.xlsx"
TOP_K = 5


def normalize_source(value):
    value = str(value or "").replace("\\", "/")
    return os.path.basename(value).strip().lower()


# =========================================================
# 1. NẠP VECTOR DB RIÊNG
# =========================================================

print("🔄 Nạp VectorDB của tập dữ liệu độc lập...")

db = FAISS.load_local(
    DB_PATH,
    vectordb.load_embedding(),
    allow_dangerous_deserialization=True,
)

# Cho Hybrid Search của production dùng DB thử nghiệm này.
llm_chain.load_vector_db = lambda: db

# Reset BM25 cache để chắc chắn BM25 được tạo từ DB mới.
llm_chain._hybrid_retriever_cache = {
    "bm25": None,
    "valid_docs": None,
    "doc_to_index": None,
}

print("✅ Đã nạp VectorDB riêng.")


# =========================================================
# 2. ĐỌC 20 CÂU HỎI
# =========================================================

df = pd.read_excel(
    TEST_FILE,
    sheet_name="Generalization_Dataset",
)

QUESTION_COL = "Câu hỏi người dùng (Input)"
SOURCE_COL = "Nguồn (Source)"

df = df.dropna(
    subset=[QUESTION_COL, SOURCE_COL]
).reset_index(drop=True)

print(f"📋 Số câu hỏi: {len(df)}")


# =========================================================
# 3. RETRIEVAL
# =========================================================

results = []

for index, row in df.iterrows():

    question = str(row[QUESTION_COL]).strip()
    expected_source = normalize_source(row[SOURCE_COL])

    start = time.perf_counter()

    # Giống production:
    # Hybrid Search -> tối đa 25 ứng viên
    candidates = llm_chain._adaptive_hybrid_search(
        question,
        candidate_k=25,
    )

    # hybrid_only -> lấy Top-5, không Cross-Encoder
    top_docs = candidates[:TOP_K]

    latency = time.perf_counter() - start

    first_rank = None
    retrieved_sources = []

    for rank, doc in enumerate(top_docs, start=1):

        metadata = doc.metadata or {}

        source = normalize_source(
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("title")
        )

        retrieved_sources.append(source)

        if source == expected_source and first_rank is None:
            first_rank = rank

    results.append(
        {
            "index": index + 1,
            "question": question,
            "expected_source": expected_source,
            "first_gold_rank": first_rank or 0,
            "hit@1": int(first_rank == 1),
            "hit@3": int(
                first_rank is not None and first_rank <= 3
            ),
            "hit@5": int(
                first_rank is not None and first_rank <= 5
            ),
            "rr@5": (
                1.0 / first_rank
                if first_rank is not None and first_rank <= 5
                else 0.0
            ),
            "latency_s": latency,
            "retrieved_sources": " | ".join(retrieved_sources),
        }
    )

    print(
        f"{index + 1:02d}/{len(df)} | "
        f"rank={first_rank or '-'} | "
        f"source={expected_source} | "
        f"{latency:.3f}s"
    )


# =========================================================
# 4. TỔNG HỢP
# =========================================================

result_df = pd.DataFrame(results)

hit1 = result_df["hit@1"].mean()
hit3 = result_df["hit@3"].mean()
hit5 = result_df["hit@5"].mean()
mrr5 = result_df["rr@5"].mean()
avg_latency = result_df["latency_s"].mean()

output_path = (
    PROJECT_ROOT
    / "evaluation"
    / "external_kb_retrieval_results.csv"
)

result_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n========== EXTERNAL KB RETRIEVAL ==========")
print(f"Queries : {len(result_df)}")
print(f"Hit@1   : {hit1 * 100:.2f}%")
print(f"Hit@3   : {hit3 * 100:.2f}%")
print(f"Hit@5   : {hit5 * 100:.2f}%")
print(f"MRR@5   : {mrr5:.4f}")
print(f"Latency : {avg_latency:.4f}s/query")
print(f"CSV     : {output_path}")
print("===========================================")