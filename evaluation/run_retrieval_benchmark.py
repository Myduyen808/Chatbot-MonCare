import os
import sys
import csv
import json
import time
import statistics
from contextlib import redirect_stdout
from io import StringIO


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from llm_chain import _adaptive_hybrid_search
from vectordb import get_source_authority


TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "regression_cases.json",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "retrieval_benchmark_results.csv",
)

TOP_K = 5
CANDIDATE_K = 25


def normalize_source(source):
    """Chuẩn hóa tên source cho cả Windows/Linux."""

    value = str(source or "").replace(
        "\\",
        "/"
    )

    return value.rsplit("/", 1)[-1].casefold()


def get_doc_source(doc):
    metadata = doc.metadata or {}

    return metadata.get(
        "source",
        ""
    )


def get_doc_tier(doc):
    metadata = doc.metadata or {}

    return metadata.get(
        "authority_tier",
        "C"
    )


# =========================================================
# LOAD TEST CASES
# =========================================================

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy: {TEST_FILE}"
    )


with open(
    TEST_FILE,
    "r",
    encoding="utf-8",
) as f:
    cases = json.load(f)


# Chỉ tính Retrieval Hit cho case có nguồn mong đợi.
retrieval_cases = [
    case
    for case in cases
    if str(
        case.get("expected_source", "")
    ).strip()
]


if not retrieval_cases:
    raise RuntimeError(
        "Không có test case nào có expected_source."
    )


print(
    f"📋 Retrieval benchmark: "
    f"{len(retrieval_cases)} cases"
)


# =========================================================
# WARM-UP
# =========================================================

print("🔥 Warm-up FAISS + BM25...")

with redirect_stdout(StringIO()):
    _adaptive_hybrid_search(
        "Trẻ 8 tháng tuổi có nên ăn dặm không?",
        candidate_k=CANDIDATE_K,
    )

print("✅ Warm-up xong.\n")


# =========================================================
# BENCHMARK
# =========================================================

results = []

for case in retrieval_cases:

    case_id = case["id"]
    question = case["question"]
    expected_source = case["expected_source"]

    expected_normalized = normalize_source(
        expected_source
    )

    start = time.perf_counter()

    # Tắt debug dài khi benchmark.
    with redirect_stdout(StringIO()):
        docs = _adaptive_hybrid_search(
            question,
            candidate_k=CANDIDATE_K,
        )

    latency = time.perf_counter() - start

    top_docs = docs[:TOP_K]

    expected_rank = None

    for rank, doc in enumerate(
        top_docs,
        start=1,
    ):
        actual_source = normalize_source(
            get_doc_source(doc)
        )

        if actual_source == expected_normalized:
            expected_rank = rank
            break

    hit_at_5 = (
        expected_rank is not None
        and expected_rank <= 5
    )

    reciprocal_rank = (
        1.0 / expected_rank
        if expected_rank is not None
        else 0.0
    )

    expected_tier, _ = get_source_authority(
        expected_source
    )

    # Nguồn chính thống mong đợi có lọt Top-3 không?
    official_hit_at_3 = None

    if expected_tier == "A":
        official_hit_at_3 = (
            expected_rank is not None
            and expected_rank <= 3
        )

    # Trong Top-3 có ít nhất một Tier-A hay không?
    tier_a_in_top3 = any(
        get_doc_tier(doc) == "A"
        for doc in docs[:3]
    )

    results.append(
        {
            "id": case_id,
            "group": case.get(
                "group",
                ""
            ),
            "question": question,
            "expected_source": expected_source,
            "expected_tier": expected_tier,
            "rank": (
                expected_rank
                if expected_rank is not None
                else ""
            ),
            "hit_at_5": int(hit_at_5),
            "reciprocal_rank": reciprocal_rank,
            "official_hit_at_3": (
                ""
                if official_hit_at_3 is None
                else int(official_hit_at_3)
            ),
            "tier_a_in_top3": int(
                tier_a_in_top3
            ),
            "latency_seconds": latency,
        }
    )

    rank_display = (
        expected_rank
        if expected_rank is not None
        else "MISS"
    )

    print(
        f"{case_id} | "
        f"rank={rank_display} | "
        f"Hit@5={hit_at_5} | "
        f"Tier={expected_tier} | "
        f"{latency:.3f}s"
    )


# =========================================================
# METRICS
# =========================================================

total = len(results)

hit5 = sum(
    row["hit_at_5"]
    for row in results
)

hit_at_5_score = (
    hit5 / total
    if total
    else 0.0
)

mrr_at_5 = (
    sum(
        row["reciprocal_rank"]
        for row in results
    )
    / total
    if total
    else 0.0
)

official_rows = [
    row
    for row in results
    if row["expected_tier"] == "A"
]

official_hit3 = (
    sum(
        int(row["official_hit_at_3"])
        for row in official_rows
    )
    / len(official_rows)
    if official_rows
    else 0.0
)

tier_a_top3 = (
    sum(
        row["tier_a_in_top3"]
        for row in results
    )
    / total
    if total
    else 0.0
)

latencies = [
    row["latency_seconds"]
    for row in results
]

avg_latency = (
    statistics.mean(latencies)
    if latencies
    else 0.0
)


# =========================================================
# SAVE CSV
# =========================================================

fieldnames = list(
    results[0].keys()
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n========== RETRIEVAL BENCHMARK ==========")

print(
    f"Cases: {total}"
)

print(
    f"Hit@5: "
    f"{hit_at_5_score:.2%}"
)

print(
    f"MRR@5: "
    f"{mrr_at_5:.4f}"
)

print(
    f"Official Hit@3: "
    f"{official_hit3:.2%}"
)

print(
    f"Tier-A in Top-3: "
    f"{tier_a_top3:.2%}"
)

print(
    f"Average retrieval latency: "
    f"{avg_latency:.3f}s"
)

print(
    f"CSV: {OUTPUT_FILE}"
)

print("========================================")