"""Chạy RAGAS riêng từng bộ 100 câu KB1 hoặc KB3.

Ví dụ:
    python run_ragas_100_each.py --kb kb1 --batch 1
    python run_ragas_100_each.py --kb kb3 --batch 1

Tên CSV mặc định:
    kb1_batch_1.csv
    kb3_batch_1.csv

Nếu tên file khác, truyền rõ đường dẫn bằng --input.
Mỗi KB có thư mục checkpoint/kết quả riêng nên không ghi đè nhau.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

try:
    import torch
except ImportError:  # Torch chỉ dùng để dọn CUDA; không bắt buộc khi chạy CPU.
    torch = None


warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
load_dotenv(override=False)

METRIC_NAMES = [
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "context_precision",
]

ERROR_PATTERNS = [
    "lỗi llm",
    "error code",
    "rate limit",
    "429",
    "không thể kết nối",
    "không tìm thấy thông tin trong tài liệu",
]

INVALID_ANSWER_PATTERNS = [
    "không tìm thấy thông tin",
    "không thể hỗ trợ yêu cầu này",
    "momcare không thể",
    "đưa bé đến cơ sở y tế để được thăm khám",
    "hệ thống ai đang quá tải",
    "không có câu trả lời",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy RAGAS tách riêng 100 câu KB1 hoặc 100 câu KB3."
    )
    parser.add_argument("--kb", choices=("kb1", "kb3"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV đầu vào; mặc định là <kb>_batch_<batch>.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Số câu cần chạy; mặc định và khuyến nghị là 100",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Vị trí bắt đầu lấy dữ liệu, mặc định 0",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("ragas_runs"),
        help="Thư mục gốc chứa kết quả",
    )
    parser.add_argument("--k", type=int, default=5, help="Số tài liệu truy xuất")
    parser.add_argument("--rag-delay", type=float, default=3.0)
    parser.add_argument("--eval-delay", type=float, default=10.0)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Bỏ checkpoint cũ và chạy lại từ đầu cho đúng KB/batch này",
    )
    return parser.parse_args()


def validate_and_select(df: pd.DataFrame, *, offset: int, limit: int) -> pd.DataFrame:
    required = {"question", "ground_truth"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV thiếu cột bắt buộc: " + ", ".join(sorted(missing))
        )
    if offset < 0 or limit <= 0:
        raise ValueError("--offset phải >= 0 và --limit phải > 0")
    if len(df) < offset + limit:
        raise ValueError(
            f"CSV chỉ có {len(df)} dòng, không đủ lấy {limit} câu "
            f"từ vị trí {offset}."
        )

    selected = df.iloc[offset : offset + limit].copy().reset_index(drop=True)
    selected.insert(0, "row_id", range(offset, offset + limit))
    selected["question"] = selected["question"].fillna("").astype(str).str.strip()
    selected["ground_truth"] = (
        selected["ground_truth"].fillna("").astype(str).str.strip()
    )
    if (selected["question"] == "").any():
        bad = selected.loc[selected["question"] == "", "row_id"].tolist()
        raise ValueError(f"Có câu hỏi rỗng tại row_id: {bad}")
    return selected


def is_valid_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    lowered = answer.lower().strip()
    if len(lowered) < 15:
        return False
    return not any(pattern in lowered for pattern in ERROR_PATTERNS)


def is_valid_for_report(answer: str) -> bool:
    if not is_valid_answer(answer):
        return False
    lowered = answer.lower().strip()
    return not any(pattern in lowered for pattern in INVALID_ANSWER_PATTERNS)


def document_text(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("content") or "")
    return str(doc)


def document_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None)
    if metadata is None and isinstance(doc, dict):
        metadata = doc.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("source", "N/A"))
    return "N/A"


def save_checkpoint(records: dict[int, dict[str, Any]], path: Path) -> None:
    checkpoint_df = pd.DataFrame(records.values())
    if not checkpoint_df.empty and "row_id" in checkpoint_df.columns:
        checkpoint_df = checkpoint_df.sort_values("row_id")
    checkpoint_df.to_csv(path, index=False, encoding="utf-8-sig")


def load_checkpoint(path: Path, selected: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    checkpoint = pd.read_csv(path, encoding="utf-8-sig")
    if checkpoint.empty:
        return {}
    if "row_id" not in checkpoint.columns or "question" not in checkpoint.columns:
        raise ValueError(
            f"Checkpoint {path} không đúng định dạng. Dùng --restart để chạy mới."
        )

    expected = selected.set_index("row_id")["question"].to_dict()
    records: dict[int, dict[str, Any]] = {}
    for record in checkpoint.to_dict("records"):
        row_id = int(record["row_id"])
        if row_id not in expected:
            continue
        if str(record.get("question", "")).strip() != expected[row_id]:
            raise ValueError(
                "Checkpoint không khớp CSV hiện tại tại "
                f"row_id={row_id}. Dùng --restart nếu muốn chạy lại."
            )
        records[row_id] = record
    return records


def result_is_complete(record: dict[str, Any] | None) -> bool:
    if not record or record.get("status") != "done":
        return False
    return all(pd.notna(record.get(metric)) for metric in METRIC_NAMES)


def rate_limit_wait(error: str, attempt: int) -> float:
    match = re.search(r"in (\d+)m([\d.]+)s", error)
    if match:
        return int(match.group(1)) * 60 + float(match.group(2)) + 15
    seconds_match = re.search(r"try again in ([\d.]+)s", error.lower())
    if seconds_match:
        return float(seconds_match.group(1)) + 15
    return 90.0 * (attempt + 1)


def metric_number(value: Any) -> float:
    if hasattr(value, "value"):
        value = value.value
    return float(value)


def call_collection_metric(metric: Any, **kwargs: Any) -> float:
    if hasattr(metric, "score"):
        return metric_number(metric.score(**kwargs))
    return metric_number(asyncio.run(metric.ascore(**kwargs)))


def main() -> None:
    args = parse_args()
    kb_name = args.kb.lower()
    input_file = args.input or Path(f"{kb_name}_batch_{args.batch}.csv")
    output_dir = args.output_root / kb_name / f"batch_{args.batch}_{args.limit}_cau"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "checkpoint.csv"
    output_file = output_dir / f"result_{kb_name}_{args.limit}_cau.csv"
    summary_file = output_dir / "summary.json"

    if args.restart:
        for path in (checkpoint_file, output_file, summary_file):
            if path.exists():
                path.unlink()

    print("\n" + "=" * 72)
    print(f"RAGAS 4 METRICS — {kb_name.upper()} — {args.limit} CÂU")
    print(f"Input      : {input_file}")
    print(f"Checkpoint : {checkpoint_file}")
    print(f"Output     : {output_file}")
    print("=" * 72 + "\n")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {input_file}. Đặt CSV cạnh script hoặc dùng --input."
        )

    source_df = pd.read_csv(input_file, encoding="utf-8-sig")
    selected = validate_and_select(
        source_df, offset=args.offset, limit=args.limit
    )
    print(f"✅ Đã khóa đúng {len(selected)} câu từ {input_file}")

    judge_keys = [
        key
        for key in (
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        )
        if key
    ]
    if not judge_keys:
        raise ValueError("Không có GROQ_API_KEY trong biến môi trường hoặc .env")
    print(f"✅ Có {len(judge_keys)} Groq key(s)")

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # Ragas 0.3 và một số bản 0.4 còn giữ API legacy. Bản 0.4 mới dùng
    # ragas.metrics.collections; script hỗ trợ cả hai nhánh.
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig

        ragas_mode = "legacy-evaluate"
    except ImportError:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        ragas_mode = "v0.4-collections"

    print(f"✅ Chế độ Ragas: {ragas_mode}")
    print("✅ Metrics: Faithfulness | ContextRecall | AnswerRelevancy | ContextPrecision")

    def get_judge_llm() -> Any:
        return LangchainLLMWrapper(
            ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=random.choice(judge_keys),
            )
        )

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    def evaluate_one(item: dict[str, Any], max_retry: int = 4) -> dict[str, float] | None:
        for attempt in range(max_retry):
            try:
                llm = get_judge_llm()
                metrics = [
                    Faithfulness(llm=llm),
                    ContextRecall(llm=llm),
                    AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
                    ContextPrecision(llm=llm),
                ]

                if ragas_mode == "legacy-evaluate":
                    dataset = Dataset.from_dict(
                        {
                            "question": [item["question"]],
                            "answer": [item["answer"]],
                            "contexts": [item["contexts"]],
                            "ground_truth": [item["ground_truth"]],
                        }
                    )
                    evaluated = evaluate(
                        dataset=dataset,
                        metrics=metrics,
                        raise_exceptions=False,
                        run_config=RunConfig(
                            max_workers=1, timeout=300, max_retries=2
                        ),
                    )
                    row = evaluated.to_pandas().iloc[0].to_dict()
                    return {
                        name: float(row[name])
                        for name in METRIC_NAMES
                        if pd.notna(row.get(name))
                    }

                return {
                    "faithfulness": call_collection_metric(
                        metrics[0],
                        user_input=item["question"],
                        response=item["answer"],
                        retrieved_contexts=item["contexts"],
                    ),
                    "context_recall": call_collection_metric(
                        metrics[1],
                        user_input=item["question"],
                        reference=item["ground_truth"],
                        retrieved_contexts=item["contexts"],
                    ),
                    "answer_relevancy": call_collection_metric(
                        metrics[2],
                        user_input=item["question"],
                        response=item["answer"],
                    ),
                    "context_precision": call_collection_metric(
                        metrics[3],
                        user_input=item["question"],
                        reference=item["ground_truth"],
                        retrieved_contexts=item["contexts"],
                    ),
                }
            except Exception as exc:
                error = str(exc)
                if "429" in error or "rate limit" in error.lower():
                    wait = rate_limit_wait(error, attempt)
                    print(f"\n  ⏳ Rate limit — chờ {wait:.0f}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"\n  ⚠️ Lỗi chấm: {error[:120]}", flush=True)
                    time.sleep(15)
        return None

    from llm_chain import RAGChain

    chain = RAGChain(k=args.k)
    records = load_checkpoint(checkpoint_file, selected)
    completed = sum(result_is_complete(record) for record in records.values())
    if records:
        print(
            f"⚡ Checkpoint có {len(records)} dòng; "
            f"{completed} câu đã đủ 4 metrics và sẽ được bỏ qua."
        )

    total = len(selected)
    for position, source_row in selected.iterrows():
        row_id = int(source_row["row_id"])
        question = source_row["question"]
        ground_truth = source_row["ground_truth"]

        if result_is_complete(records.get(row_id)):
            print(f"[{position + 1:>3}/{total}] SKIP — đã chấm xong")
            continue

        print("\n" + "─" * 72)
        print(f"[{position + 1:>3}/{total}] row_id={row_id}: {question[:80]}")
        base_record: dict[str, Any] = {
            "row_id": row_id,
            "kb": kb_name,
            "question": question,
            "ground_truth": ground_truth,
        }

        try:
            response = chain.invoke({"question": question, "history": []})
            answer = str(response.get("answer", "")).strip()
            docs = response.get("docs", []) or []

            print(f"  📄 Retrieved docs: {len(docs)}")
            for index, doc in enumerate(docs):
                preview = document_text(doc)[:120].replace("\n", " ")
                print(
                    f"     [{index + 1}] {document_source(doc)} -> {preview}..."
                )

            contexts = [document_text(doc)[:1200] for doc in docs]
            contexts = [text for text in contexts if text.strip()]
            base_record.update(
                {
                    "answer": answer,
                    "contexts": contexts,
                    "retrieved_docs": len(docs),
                }
            )

            if not is_valid_answer(answer) or not contexts:
                base_record["status"] = "invalid_rag"
                base_record["error"] = "Câu trả lời hoặc ngữ cảnh không hợp lệ"
                for metric in METRIC_NAMES:
                    base_record[metric] = None
                print("  ⚠️ Không chấm RAGAS vì answer/context không hợp lệ")
            else:
                print(f"  🤖 Answer: {answer[:160]}...")
                scores = evaluate_one(base_record)
                if scores and all(name in scores for name in METRIC_NAMES):
                    base_record.update(scores)
                    base_record["status"] = "done"
                    base_record["error"] = ""
                    print(
                        "  ✅ "
                        + " | ".join(
                            f"{name}={scores[name]:.3f}"
                            for name in METRIC_NAMES
                        )
                    )
                else:
                    base_record["status"] = "eval_error"
                    base_record["error"] = "Không chấm đủ 4 metrics sau retry"
                    for metric in METRIC_NAMES:
                        base_record.setdefault(metric, None)
                    print("  ❌ Không chấm đủ 4 metrics")
        except Exception as exc:
            base_record["status"] = "rag_error"
            base_record["error"] = str(exc)[:500]
            base_record.setdefault("answer", "")
            base_record.setdefault("contexts", [])
            for metric in METRIC_NAMES:
                base_record[metric] = None
            print(f"  ❌ Lỗi RAG: {str(exc)[:160]}")

        base_record["is_valid_rag"] = is_valid_for_report(
            str(base_record.get("answer", ""))
        )
        records[row_id] = base_record
        save_checkpoint(records, checkpoint_file)

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(args.eval_delay if base_record["status"] == "done" else args.rag_delay)

    # Luôn xuất đủ đúng 100 dòng đã chọn, kể cả câu lỗi để dễ kiểm đếm.
    final_rows = []
    for _, source_row in selected.iterrows():
        row_id = int(source_row["row_id"])
        record = records.get(
            row_id,
            {
                "row_id": row_id,
                "kb": kb_name,
                "question": source_row["question"],
                "ground_truth": source_row["ground_truth"],
                "status": "not_run",
                "error": "Chưa chạy",
            },
        )
        final_rows.append(record)

    final_df = pd.DataFrame(final_rows).sort_values("row_id")
    for metric in METRIC_NAMES:
        if metric not in final_df.columns:
            final_df[metric] = None
        final_df[metric] = pd.to_numeric(final_df[metric], errors="coerce").round(3)
    if "is_valid_rag" not in final_df.columns:
        final_df["is_valid_rag"] = False

    column_order = [
        "row_id",
        "kb",
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "retrieved_docs",
        *METRIC_NAMES,
        "is_valid_rag",
        "status",
        "error",
    ]
    final_df = final_df[
        [column for column in column_order if column in final_df.columns]
    ]
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    done_df = final_df[final_df["status"] == "done"]
    valid_df = done_df[done_df["is_valid_rag"] == True]
    summary = {
        "kb": kb_name,
        "input_file": str(input_file),
        "requested_questions": total,
        "completed_4_metrics": int(len(done_df)),
        "valid_for_report": int(len(valid_df)),
        "failed_or_pending": int(total - len(done_df)),
        "mean_all_completed": {
            metric: (
                round(float(done_df[metric].mean()), 3)
                if len(done_df) and done_df[metric].notna().any()
                else None
            )
            for metric in METRIC_NAMES
        },
        "mean_valid_only": {
            metric: (
                round(float(valid_df[metric].mean()), 3)
                if len(valid_df) and valid_df[metric].notna().any()
                else None
            )
            for metric in METRIC_NAMES
        },
        "result_file": str(output_file),
        "checkpoint_file": str(checkpoint_file),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"KẾT QUẢ {kb_name.upper()} — yêu cầu {total} câu")
    print(f"Đủ 4 metrics : {len(done_df)}/{total}")
    print(f"Hợp lệ báo cáo: {len(valid_df)}/{total}")
    print(f"Kết quả       : {output_file}")
    print(f"Tóm tắt       : {summary_file}")
    print("=" * 72)


if __name__ == "__main__":
    main()
