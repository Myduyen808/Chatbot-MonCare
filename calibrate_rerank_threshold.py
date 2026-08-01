"""
Hiệu chỉnh ngưỡng RERANK_MIN_SCORE cho MomCare.

Mục tiêu
--------
1. Đọc 3 bộ kiểm thử KB1, KB2, KB3.
2. Chạy đúng phần Retrieval + Cross-Encoder của hệ thống, KHÔNG gọi LLM sinh câu trả lời.
3. Ghi điểm Cross-Encoder cao nhất cho từng câu hỏi.
4. Xem tài liệu Top-1 có đúng nguồn Ground Truth hay không.
5. Tìm ngưỡng giúp hạn chế chấp nhận tài liệu sai nhưng vẫn giữ Recall tối thiểu.

Lưu ý quan trọng
----------------
- Điểm của cross-encoder/ms-marco-MiniLM-L-6-v2 là logit, KHÔNG phải xác suất 0..1.
- Vì vậy không được tự đặt 0.65 nếu chưa chạy thực nghiệm.
- Ba file KB hiện chủ yếu là câu RAG có nguồn đúng. Nhãn âm trong thí nghiệm này
  được hình thành từ các trường hợp hệ thống truy xuất Top-1 sai nguồn.
- Muốn hiệu chỉnh cơ chế từ chối ngoài miền chặt chẽ hơn, nên bổ sung thêm một tập
  câu hỏi không có câu trả lời trong kho tri thức.

Cách chạy
---------
Đặt file này cùng thư mục với llm_chain.py, vectordb.py, db_config.yml,
model_config.yml và VectorDB hiện tại, sau đó chạy:

    python calibrate_rerank_threshold.py

Hoặc chỉ chạy một phần dữ liệu để thử nhanh:

    python calibrate_rerank_threshold.py --limit-per-file 50

Kết quả được lưu trong thư mục rerank_threshold_results/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Import đúng logic từ hệ thống hiện tại.
import llm_chain
from vectordb import smart_retrieve


DEFAULT_INPUTS = [
    ("KB1", "KB1_Medical_Standard.xlsx"),
    ("KB2", "KB2_Mom_Style.xlsx"),
    ("KB3", "KB3_Information_Noise.xlsx"),
]

QUESTION_COLUMN = "Câu hỏi người dùng (Input)"
EXPECTED_SOURCE_COLUMN = "Nguồn (Source)"
GT_COLUMN = "Nhãn (GT)"
EXPECTED_ANSWER_COLUMN = "Phản hồi kỳ vọng (Expected Output)"


@dataclass
class EvaluationRow:
    scenario: str
    row_id: int
    question: str
    expected_source: str
    expected_answer: str
    predicted_top1_source: str
    top1_source_match: int
    expected_source_in_top5: int
    best_rerank_score: float
    expected_source_best_score: Optional[float]
    candidate_count: int
    elapsed_seconds: float
    error: str = ""


def normalize_source(value: object) -> str:
    """Chuẩn hóa tên nguồn để so sánh ổn định giữa đường dẫn và tên file."""
    if value is None:
        return ""
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return ""
    return os.path.basename(text).strip().casefold()


def normalize_text(value: object) -> str:
    """Chuẩn hóa tiếng Việt để so khớp nội dung tham chiếu với chunk."""
    text = str(value or "").casefold()
    text = re.sub(r"[^0-9a-zà-ỹ\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_coverage(expected_answer: str, document_text: str) -> float:
    """Tỷ lệ token của đáp án kỳ vọng xuất hiện trong tài liệu."""
    expected_tokens = normalize_text(expected_answer).split()
    document_tokens = set(normalize_text(document_text).split())
    if not expected_tokens:
        return 0.0
    return sum(token in document_tokens for token in expected_tokens) / len(expected_tokens)


def is_relevant_document(doc, expected_source: str, expected_answer: str) -> bool:
    """Đúng nếu khớp nguồn HOẶC nội dung bao phủ đáp án kỳ vọng."""
    source_match = (
        bool(normalize_source(expected_source))
        and normalize_source(doc.metadata.get("source", "")) == normalize_source(expected_source)
    )
    content_match = answer_coverage(expected_answer, doc.page_content) >= 0.70
    return source_match or content_match


def document_key(doc) -> str:
    """Khóa khử trùng lặp tương tự pipeline chính."""
    normalized = re.sub(r"\s+", " ", str(doc.page_content or "")).strip()
    return normalized[:500]


def build_candidate_pool(question: str, use_multi_query: bool) -> list:
    """
    Mô phỏng phần tạo tập ứng viên trong RAGChain.invoke().

    - Hybrid lấy 25 ứng viên.
    - Có thể bổ sung tối đa 2 truy vấn mở rộng cho câu hỏi <= 5 từ.
    - Khử trùng lặp.
    - Giới hạn tối đa 40 ứng viên trước Cross-Encoder.
    """
    primary_docs = llm_chain._adaptive_hybrid_search(
        question,
        candidate_k=llm_chain.FAISS_CANDIDATE_K,
    )

    all_docs: list = []
    seen: set[str] = set()

    def add_unique(docs: Iterable) -> None:
        for doc in docs:
            key = document_key(doc)
            if key and key not in seen:
                seen.add(key)
                all_docs.append(doc)

    add_unique(primary_docs)

    if use_multi_query and len(question.split()) <= 5:
        extra_queries = llm_chain.generate_multi_queries(question, n=2)
        for expanded_query in extra_queries[1:]:
            add_unique(smart_retrieve(expanded_query, None, 10))

    return all_docs[: llm_chain.MAX_RERANK_CANDIDATES]


def evaluate_question(
    scenario: str,
    row_id: int,
    question: str,
    expected_source: str,
    expected_answer: str,
    use_multi_query: bool,
) -> EvaluationRow:
    started = time.perf_counter()

    try:
        candidates = build_candidate_pool(question, use_multi_query)
        if not candidates:
            return EvaluationRow(
                scenario=scenario,
                row_id=row_id,
                question=question,
                expected_source=expected_source,
                expected_answer=expected_answer,
                predicted_top1_source="",
                top1_source_match=0,
                expected_source_in_top5=0,
                best_rerank_score=float("nan"),
                expected_source_best_score=None,
                candidate_count=0,
                elapsed_seconds=time.perf_counter() - started,
                error="Không có tài liệu ứng viên",
            )

        reranker = llm_chain.get_reranker()
        pairs = [(question, doc.page_content) for doc in candidates]
        scores = reranker.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(scores, candidates),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        best_score = float(ranked[0][0])
        top1_source = normalize_source(ranked[0][1].metadata.get("source", ""))
        top1_is_relevant = is_relevant_document(
            ranked[0][1], expected_source, expected_answer
        )

        top5_is_relevant = any(
            is_relevant_document(doc, expected_source, expected_answer)
            for _, doc in ranked[: llm_chain.DEFAULT_TOP_K]
        )

        matching_scores = [
            float(score)
            for score, doc in ranked
            if is_relevant_document(doc, expected_source, expected_answer)
        ]

        return EvaluationRow(
            scenario=scenario,
            row_id=row_id,
            question=question,
            expected_source=expected_source,
            expected_answer=expected_answer,
            predicted_top1_source=top1_source,
            top1_source_match=int(top1_is_relevant),
            expected_source_in_top5=int(top5_is_relevant),
            best_rerank_score=best_score,
            expected_source_best_score=max(matching_scores) if matching_scores else None,
            candidate_count=len(candidates),
            elapsed_seconds=time.perf_counter() - started,
        )

    except Exception as exc:  # Tiếp tục chạy dù một mẫu lỗi.
        return EvaluationRow(
            scenario=scenario,
            row_id=row_id,
            question=question,
            expected_source=expected_source,
            expected_answer=expected_answer,
            predicted_top1_source="",
            top1_source_match=0,
            expected_source_in_top5=0,
            best_rerank_score=float("nan"),
            expected_source_best_score=None,
            candidate_count=0,
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metric_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """
    label = 1: tài liệu Top-1 đúng nguồn.
    accept = 1: best_rerank_score >= threshold.

    FP là trường hợp nguy hiểm cần hạn chế: Top-1 sai nhưng vẫn được chấp nhận.
    """
    accepted = scores >= threshold
    positive = labels == 1

    tp = int(np.sum(accepted & positive))
    fp = int(np.sum(accepted & ~positive))
    fn = int(np.sum(~accepted & positive))
    tn = int(np.sum(~accepted & ~positive))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    false_accept_rate = safe_div(fp, fp + tn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, len(labels))
    coverage = safe_div(tp + fp, len(labels))

    return {
        "threshold": float(threshold),
        "tp_correct_accept": tp,
        "fp_wrong_accept": fp,
        "fn_correct_reject": fn,
        "tn_wrong_reject": tn,
        "precision_when_answering": precision,
        "recall_of_correct_retrieval": recall,
        "specificity_wrong_retrieval": specificity,
        "false_accept_rate": false_accept_rate,
        "f1": f1,
        "accuracy": accuracy,
        "coverage_answer_rate": coverage,
    }


def generate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Tạo các ngưỡng trên đúng miền logit quan sát được."""
    unique_scores = np.unique(scores)
    if len(unique_scores) <= 500:
        mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        thresholds = np.concatenate(
            ([unique_scores.min() - 1e-6], mids, [unique_scores.max() + 1e-6])
        )
    else:
        thresholds = np.quantile(scores, np.linspace(0, 1, 501))
        thresholds = np.unique(thresholds)
    return thresholds


def choose_threshold(metrics_df: pd.DataFrame, minimum_recall: float) -> tuple[pd.Series, str]:
    """
    Chính sách ưu tiên an toàn:
    1. Chỉ xét các ngưỡng giữ Recall >= minimum_recall.
    2. Trong nhóm đó, giảm False Accept Rate trước.
    3. Nếu hòa, ưu tiên Precision rồi Coverage.
    4. Nếu không có ngưỡng đạt Recall yêu cầu, dùng ngưỡng F1 tốt nhất.
    """
    eligible = metrics_df[
        metrics_df["recall_of_correct_retrieval"] >= minimum_recall
    ].copy()

    if not eligible.empty:
        chosen = eligible.sort_values(
            by=[
                "false_accept_rate",
                "precision_when_answering",
                "coverage_answer_rate",
                "threshold",
            ],
            ascending=[True, False, False, False],
        ).iloc[0]
        policy = (
            f"Ưu tiên giảm chấp nhận sai với Recall >= {minimum_recall:.2f}."
        )
        return chosen, policy

    chosen = metrics_df.sort_values(
        by=["f1", "precision_when_answering", "recall_of_correct_retrieval"],
        ascending=[False, False, False],
    ).iloc[0]
    return chosen, "Không có ngưỡng đạt Recall yêu cầu; chọn F1 cao nhất."


def summarize_score_distribution(df: pd.DataFrame) -> dict:
    summary: dict = {}
    for label_value, name in [(1, "top1_correct"), (0, "top1_wrong")]:
        values = df.loc[
            df["top1_source_match"] == label_value,
            "best_rerank_score",
        ].dropna()
        if values.empty:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(values.count()),
            "min": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return summary


def load_dataset(path: Path, limit: Optional[int]) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    required = {QUESTION_COLUMN, EXPECTED_SOURCE_COLUMN, EXPECTED_ANSWER_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} thiếu cột: {sorted(missing)}")

    frame = frame.dropna(subset=[QUESTION_COLUMN, EXPECTED_SOURCE_COLUMN]).copy()
    if GT_COLUMN in frame.columns:
        frame = frame[frame[GT_COLUMN].astype(str).str.upper().eq("RAG")]

    if limit is not None:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hiệu chỉnh RERANK_MIN_SCORE từ KB1, KB2, KB3."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Thư mục chứa ba file Excel (mặc định: thư mục hiện tại).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rerank_threshold_results"),
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Chạy thử N mẫu đầu mỗi file; bỏ trống để chạy toàn bộ.",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.90,
        help="Recall tối thiểu khi chọn ngưỡng (mặc định 0.90).",
    )
    parser.add_argument(
        "--use-multi-query",
        action="store_true",
        help="Bật Multi-Query cho câu <=5 từ; có thể phát sinh gọi Groq API.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Ghi checkpoint sau mỗi N mẫu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < args.minimum_recall <= 1:
        raise ValueError("--minimum-recall phải nằm trong (0, 1].")

    all_rows: list[EvaluationRow] = []
    total_processed = 0

    print("=" * 72)
    print("THỰC NGHIỆM HIỆU CHỈNH RERANK_MIN_SCORE")
    print(f"Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"FAISS candidates: {llm_chain.FAISS_CANDIDATE_K}")
    print(f"Max rerank candidates: {llm_chain.MAX_RERANK_CANDIDATES}")
    print(f"Top-k cuối: {llm_chain.DEFAULT_TOP_K}")
    print(f"Multi-Query: {'BẬT' if args.use_multi_query else 'TẮT'}")
    print("=" * 72)

    for scenario, filename in DEFAULT_INPUTS:
        path = args.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy dùng --data-dir đúng thư mục dữ liệu."
            )

        frame = load_dataset(path, args.limit_per_file)
        print(f"\n[{scenario}] {filename}: {len(frame)} mẫu")

        for index, row in frame.iterrows():
            question = str(row[QUESTION_COLUMN]).strip()
            expected_source = str(row[EXPECTED_SOURCE_COLUMN]).strip()
            expected_answer = str(row[EXPECTED_ANSWER_COLUMN]).strip()

            result = evaluate_question(
                scenario=scenario,
                row_id=index + 1,
                question=question,
                expected_source=expected_source,
                expected_answer=expected_answer,
                use_multi_query=args.use_multi_query,
            )
            all_rows.append(result)
            total_processed += 1

            status = "ĐÚNG" if result.top1_source_match else "SAI"
            print(
                f"{scenario} {index + 1:03d}/{len(frame):03d} | "
                f"score={result.best_rerank_score:8.4f} | "
                f"Top1={status} | "
                f"Top5={'CÓ' if result.expected_source_in_top5 else 'KHÔNG'} | "
                f"Pred={result.predicted_top1_source[:28]} | "
                f"{result.elapsed_seconds:5.2f}s"
            )

            if total_processed % args.checkpoint_every == 0:
                checkpoint = pd.DataFrame(asdict(item) for item in all_rows)
                checkpoint.to_csv(
                    args.output_dir / "rerank_scores_checkpoint.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    details_df = pd.DataFrame(asdict(item) for item in all_rows)
    details_path = args.output_dir / "rerank_scores_detail.csv"
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    valid_df = details_df[
        details_df["error"].fillna("").eq("")
        & details_df["best_rerank_score"].notna()
    ].copy()

    if valid_df.empty:
        print("Không có mẫu hợp lệ để chọn ngưỡng.")
        return 1

    scores = valid_df["best_rerank_score"].to_numpy(dtype=float)
    labels = valid_df["top1_source_match"].to_numpy(dtype=int)

    if len(np.unique(labels)) < 2:
        warning = (
            "Dữ liệu chỉ có một lớp Top-1 đúng/sai nên không thể hiệu chỉnh "
            "ngưỡng phân biệt một cách đáng tin cậy. Hãy bổ sung mẫu truy xuất sai "
            "hoặc câu hỏi ngoài kho tri thức."
        )
        print(f"\nCẢNH BÁO: {warning}")
    else:
        warning = ""

    threshold_rows = [
        metric_at_threshold(scores, labels, threshold)
        for threshold in generate_thresholds(scores)
    ]
    metrics_df = pd.DataFrame(threshold_rows)
    metrics_path = args.output_dir / "threshold_candidates.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    chosen, policy = choose_threshold(metrics_df, args.minimum_recall)
    distribution = summarize_score_distribution(valid_df)

    per_scenario = (
        valid_df.groupby("scenario")
        .agg(
            samples=("question", "count"),
            top1_accuracy=("top1_source_match", "mean"),
            top5_hit_rate=("expected_source_in_top5", "mean"),
            mean_best_score=("best_rerank_score", "mean"),
            median_best_score=("best_rerank_score", "median"),
            mean_latency_seconds=("elapsed_seconds", "mean"),
        )
        .reset_index()
    )
    per_scenario.to_csv(
        args.output_dir / "scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    chosen_dict = {
        key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
        for key, value in chosen.to_dict().items()
    }

    summary = {
        "recommended_threshold": chosen_dict["threshold"],
        "selection_policy": policy,
        "minimum_recall_requested": args.minimum_recall,
        "warning": warning,
        "metrics_at_recommended_threshold": chosen_dict,
        "score_distribution": distribution,
        "overall": {
            "samples": int(len(valid_df)),
            "errors": int(len(details_df) - len(valid_df)),
            "top1_accuracy_without_threshold": float(valid_df["top1_source_match"].mean()),
            "top5_hit_rate_without_threshold": float(valid_df["expected_source_in_top5"].mean()),
            "mean_latency_seconds": float(valid_df["elapsed_seconds"].mean()),
        },
        "configuration": {
            "faiss_candidate_k": llm_chain.FAISS_CANDIDATE_K,
            "max_rerank_candidates": llm_chain.MAX_RERANK_CANDIDATES,
            "top_k": llm_chain.DEFAULT_TOP_K,
            "multi_query_enabled": args.use_multi_query,
        },
    }

    with open(
        args.output_dir / "threshold_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with open(
        args.output_dir / "recommended_config.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Chỉ dùng sau khi kiểm tra kết quả và chấp nhận trade-off:\n"
            f"RERANK_MIN_SCORE = {chosen_dict['threshold']:.6f}\n\n"
            f"# {policy}\n"
            f"# Precision khi trả lời: {chosen_dict['precision_when_answering']:.4f}\n"
            f"# Recall truy xuất đúng: {chosen_dict['recall_of_correct_retrieval']:.4f}\n"
            f"# False Accept Rate: {chosen_dict['false_accept_rate']:.4f}\n"
            f"# Coverage: {chosen_dict['coverage_answer_rate']:.4f}\n"
        )

    print("\n" + "=" * 72)
    print("KẾT QUẢ ĐỀ XUẤT")
    print(f"Ngưỡng logit: {chosen_dict['threshold']:.6f}")
    print(f"Chính sách: {policy}")
    print(
        f"Precision khi hệ thống quyết định trả lời: "
        f"{chosen_dict['precision_when_answering']:.4f}"
    )
    print(
        f"Recall giữ lại các trường hợp truy xuất đúng: "
        f"{chosen_dict['recall_of_correct_retrieval']:.4f}"
    )
    print(f"False Accept Rate: {chosen_dict['false_accept_rate']:.4f}")
    print(f"Coverage: {chosen_dict['coverage_answer_rate']:.4f}")
    print("\nDòng cấu hình đề xuất:")
    print(f"RERANK_MIN_SCORE = {chosen_dict['threshold']:.6f}")
    print(f"\nKết quả chi tiết: {args.output_dir.resolve()}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
