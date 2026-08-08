import sys
import re
import time
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vectordb
import llm_chain


FAISS_PATH = (
    ROOT
    / "evaluation"
    / "vimedaqa_clean"
    / "faiss_context_only"
)

QUERY_PATH = (
    ROOT
    / "evaluation"
    / "vimedaqa_clean"
    / "test_queries_and_references.csv"
)

TOP_K = 5
POOL_K = 50


# =====================================================
# LOAD BENCHMARK DB
# =====================================================

print("🔥 Loading clean ViMedAQA FAISS...")

db = FAISS.load_local(
    str(FAISS_PATH),
    vectordb.load_embedding(),
    allow_dangerous_deserialization=True,
)

all_docs = [
    db.docstore.search(doc_id)
    for doc_id in db.index_to_docstore_id.values()
]

all_docs = [
    d for d in all_docs
    if d is not None
]


def tokenize(text):
    return re.findall(
        r"[a-zA-Z0-9À-Ỹà-ỵ][a-zA-Z0-9À-Ỹà-ỵ]*",
        str(text).lower(),
    )


print("🔥 Building BM25...")

bm25 = BM25Okapi([
    tokenize(d.page_content)
    for d in all_docs
])


def key(doc):
    meta = doc.metadata or {}

    return (
        meta.get("context_hash")
        or doc.page_content[:1000]
    )


# =====================================================
# RETRIEVERS
# =====================================================

def dense_search(question, k=TOP_K):
    return db.similarity_search(
        question,
        k=k,
    )


def bm25_search(question, k=TOP_K):

    scores = bm25.get_scores(
        tokenize(question)
    )

    indexes = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )[:k]

    return [
        all_docs[i]
        for i in indexes
    ]


def hybrid_search(
    question,
    alpha,
    adaptive=False,
):

    retrieval_question = question

    if adaptive:
        retrieval_question = (
            llm_chain._expand_retrieval_query(
                question
            )
        )

    dense_docs = dense_search(
        retrieval_question,
        POOL_K,
    )

    lexical_docs = bm25_search(
        retrieval_question,
        POOL_K,
    )

    dense_rank = {
        key(doc): rank
        for rank, doc in enumerate(
            dense_docs,
            start=1,
        )
    }

    lexical_rank = {
        key(doc): rank
        for rank, doc in enumerate(
            lexical_docs,
            start=1,
        )
    }

    candidates = {}

    for doc in dense_docs + lexical_docs:
        candidates[key(doc)] = doc

    scored = []

    for doc_key, doc in candidates.items():

        vec_score = (
            1.0 / dense_rank[doc_key]
            if doc_key in dense_rank
            else 0.0
        )

        bm_score = (
            1.0 / lexical_rank[doc_key]
            if doc_key in lexical_rank
            else 0.0
        )

        score = (
            alpha * vec_score
            + (1.0 - alpha) * bm_score
        )

        scored.append(
            (score, doc)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        doc
        for _, doc in scored[:TOP_K]
    ]


def adaptive_hybrid(question):

    profile = (
        llm_chain._classify_retrieval_query(
            question
        )
    )

    alpha_map = {
        "exact_lexical": 0.20,
        "semantic": 0.30,
        "noisy_conversational": 0.30,
        "quantitative": 0.40,
    }

    alpha = alpha_map.get(
        profile,
        0.30,
    )

    return hybrid_search(
        question,
        alpha,
        adaptive=True,
    )


METHODS = {
    "Dense Only":
        lambda q: dense_search(q),

    "BM25 Only":
        lambda q: bm25_search(q),

    "Hybrid alpha=0.5":
        lambda q: hybrid_search(
            q,
            0.5,
            adaptive=False,
        ),

    "Adaptive Hybrid":
        adaptive_hybrid,
}


# =====================================================
# DATA
# =====================================================

df = pd.read_csv(
    QUERY_PATH,
    encoding="utf-8-sig",
)

print(
    f"📋 Queries: {len(df)}"
)


# Warm-up
first_question = str(
    df.iloc[0]["question"]
)

dense_search(
    first_question,
    1,
)

bm25_search(
    first_question,
    1,
)


# =====================================================
# EVALUATION
# =====================================================

all_results = []

for method_name, method in METHODS.items():

    print(
        f"\n🚀 {method_name}"
    )

    method_results = []

    for idx, row in df.iterrows():

        question = str(
            row["question"]
        )

        gold = str(
            row["gold_context_hash"]
        )

        start = time.perf_counter()

        docs = method(question)

        latency = (
            time.perf_counter()
            - start
        )

        hashes = [
            str(
                (doc.metadata or {}).get(
                    "context_hash",
                    "",
                )
            )
            for doc in docs
        ]

        rank = 0

        if gold in hashes:
            rank = (
                hashes.index(gold)
                + 1
            )

        method_results.append({
            "method": method_name,
            "row_index": row["row_index"],
            "first_gold_rank": rank,
            "hit@1": int(rank == 1),
            "hit@3": int(
                0 < rank <= 3
            ),
            "hit@5": int(
                0 < rank <= 5
            ),
            "rr@5": (
                1.0 / rank
                if 0 < rank <= 5
                else 0.0
            ),
            "latency_s": latency,
        })

        if (idx + 1) % 250 == 0:
            print(
                f"  {idx + 1}/{len(df)}"
            )

    all_results.extend(
        method_results
    )

    temp = pd.DataFrame(
        method_results
    )

    print(
        f"Hit@1={temp['hit@1'].mean()*100:.2f}% | "
        f"Hit@3={temp['hit@3'].mean()*100:.2f}% | "
        f"Hit@5={temp['hit@5'].mean()*100:.2f}% | "
        f"MRR@5={temp['rr@5'].mean():.4f} | "
        f"Latency={temp['latency_s'].mean():.4f}s"
    )


# =====================================================
# SAVE
# =====================================================

result_df = pd.DataFrame(
    all_results
)

detail_path = (
    ROOT
    / "evaluation"
    / "retrieval_ablation_clean_detail.csv"
)

summary_path = (
    ROOT
    / "evaluation"
    / "retrieval_ablation_clean_summary.csv"
)

result_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig",
)

summary = (
    result_df
    .groupby("method")
    .agg({
        "hit@1": "mean",
        "hit@3": "mean",
        "hit@5": "mean",
        "rr@5": "mean",
        "latency_s": "mean",
    })
    .reset_index()
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

print(
    "\n========== CLEAN RETRIEVAL ABLATION =========="
)

for _, row in summary.iterrows():

    print(
        f"{row['method']:<20} | "
        f"H@1={row['hit@1']*100:6.2f}% | "
        f"H@3={row['hit@3']*100:6.2f}% | "
        f"H@5={row['hit@5']*100:6.2f}% | "
        f"MRR@5={row['rr@5']:.4f} | "
        f"{row['latency_s']:.4f}s"
    )

print(
    "=============================================="
)
print("Groq calls: 0")