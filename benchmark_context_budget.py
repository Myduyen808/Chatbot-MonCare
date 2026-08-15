#!/usr/bin/env python3
"""
Context Budget Ablation cho MomCare
===================================

Mặc định: 20 câu/KB x 3 KB x 4 budget = 240 lượt RAG.
Giữ cố định Top-K=5 và toàn bộ pipeline hiện tại; chỉ thay RAG_CONTEXT_MAX_TOKENS.

Kết quả:
  context_budget_results/details.csv
  context_budget_results/summary.csv
  context_budget_results/summary_by_kb.csv
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import pandas as pd

import llm_chain
from llm_chain import RAGChain

DEFAULT_INPUTS = [
    ("KB1", "KB1_Medical_Standard.xlsx"),
    ("KB2", "KB2_Mom_Style.xlsx"),
    ("KB3", "KB3_Information_Noise.xlsx"),
]
QUESTION_COLUMN = "Câu hỏi người dùng (Input)"
ANSWER_COLUMN = "Phản hồi kỳ vọng (Expected Output)"
SOURCE_COLUMN = "Nguồn (Source)"
GT_COLUMN = "Nhãn (GT)"

INVALID_PATTERNS = [
    "không tìm thấy đủ thông tin",
    "không tìm thấy thông tin",
    "không thể hỗ trợ yêu cầu này",
    "momcare không thể",
    "hệ thống ai đang quá tải",
    "lỗi llm", "error code", "rate limit",
]

def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def is_valid_answer(answer: str) -> bool:
    text = normalize_space(answer).lower()
    if len(text) < 15:
        return False
    return not any(p in text for p in INVALID_PATTERNS)

def estimate_context_stats(docs) -> tuple[int, int]:
    total_tokens = 0
    total_chars = 0
    for idx, doc in enumerate(docs, start=1):
        safe = llm_chain.sanitize_document_text(doc.page_content)
        if not safe:
            continue
        block = f'<TAI_LIEU id="{idx}">\n{safe}\n</TAI_LIEU>'
        total_tokens += llm_chain.estimate_tokens(block)
        total_chars += len(block)
    return total_tokens, total_chars

def tokenize(text: object) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", normalize_space(text).lower())

def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]

def rouge_l_f1(prediction: str, reference: str) -> float:
    p = tokenize(prediction)
    r = tokenize(reference)
    if not p or not r:
        return 0.0
    lcs = lcs_length(p, r)
    precision = lcs / len(p)
    recall = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def load_balanced_questions(per_kb: int, seed: int) -> pd.DataFrame:
    frames = []
    for kb, file_name in DEFAULT_INPUTS:
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Không thấy {file_name}")
        df = pd.read_excel(path)
        if QUESTION_COLUMN not in df.columns:
            raise KeyError(f"{file_name} thiếu cột {QUESTION_COLUMN}")
        if ANSWER_COLUMN not in df.columns:
            raise KeyError(f"{file_name} thiếu cột {ANSWER_COLUMN}")
        work = df.copy()
        work[QUESTION_COLUMN] = work[QUESTION_COLUMN].map(normalize_space)
        work[ANSWER_COLUMN] = work[ANSWER_COLUMN].map(normalize_space)
        work = work[(work[QUESTION_COLUMN] != "") & (work[ANSWER_COLUMN] != "")]
        if GT_COLUMN in work.columns:
            gt_norm = work[GT_COLUMN].astype(str).str.strip().str.upper()
            rag_mask = gt_norm.isin({"RAG", "TRUE", "1", "YES", "ĐÚNG", "DUNG"})
            if rag_mask.any():
                work = work[rag_mask]
        n = min(per_kb, len(work))
        sampled = work.sample(n=n, random_state=seed).copy()
        sampled.insert(0, "kb", kb)
        sampled.insert(1, "source_row", sampled.index + 2)
        frames.append(sampled)
        print(f"📘 {kb}: lấy {n}/{len(work)} câu")
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "question_id", [f"Q{i:03d}" for i in range(1, len(result)+1)])
    return result

def run_experiment(questions, budgets, sleep_seconds, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    questions.to_csv(output_dir / "sampled_questions.csv", index=False, encoding="utf-8-sig")
    checkpoint = output_dir / "details.csv"

    rows = []
    done_keys = set()
    if checkpoint.exists():
        old = pd.read_csv(checkpoint, encoding="utf-8-sig")
        rows = old.to_dict("records")
        done_keys = {(str(r["question_id"]), int(r["budget"])) for r in rows}
        print(f"⚡ Resume: {len(rows)} dòng")

    chain = RAGChain(k=5)
    total = len(questions) * len(budgets)
    completed = len(done_keys)

    for _, row in questions.iterrows():
        qid = str(row["question_id"])
        question = normalize_space(row[QUESTION_COLUMN])
        reference = normalize_space(row[ANSWER_COLUMN])
        expected_source = normalize_space(row.get(SOURCE_COLUMN, ""))

        for budget in budgets:
            key = (qid, int(budget))
            if key in done_keys:
                continue

            llm_chain.RAG_CONTEXT_MAX_TOKENS = int(budget)
            started = time.perf_counter()
            error = ""
            try:
                result = chain.invoke({"question": question, "history": []})
                answer = normalize_space(result.get("answer", ""))
                docs = result.get("docs", []) or []
                retrieved_docs = result.get("retrieved_docs", []) or []
                context_tokens, context_chars = estimate_context_stats(docs)
                valid = is_valid_answer(answer)
                rouge = rouge_l_f1(answer, reference) if valid else 0.0
                sources = [normalize_space(getattr(d, "metadata", {}).get("source", "")) for d in docs]
            except Exception as exc:
                answer = ""; docs = []; retrieved_docs = []
                context_tokens = 0; context_chars = 0
                valid = False; rouge = 0.0; sources = []
                error = f"{type(exc).__name__}: {str(exc)[:300]}"

            latency = time.perf_counter() - started
            rows.append({
                "question_id": qid,
                "kb": row["kb"],
                "source_row": int(row["source_row"]),
                "budget": int(budget),
                "question": question,
                "reference_answer": reference,
                "expected_source": expected_source,
                "answer": answer,
                "valid_answer": int(valid),
                "retrieved_doc_count": len(retrieved_docs),
                "generation_doc_count": len(docs),
                "context_estimated_tokens": context_tokens,
                "context_chars": context_chars,
                "rouge_l_f1": rouge,
                "latency_seconds": latency,
                "generation_sources": " | ".join(sources),
                "error": error,
            })
            pd.DataFrame(rows).to_csv(checkpoint, index=False, encoding="utf-8-sig")
            done_keys.add(key)
            completed += 1
            print(f"[{completed:>3}/{total}] {qid} {row['kb']} budget={budget} docs={len(docs)} ctx≈{context_tokens} valid={int(valid)} ROUGE-L={rouge:.3f} {latency:.2f}s")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    details = pd.DataFrame(rows)
    summary = details.groupby("budget", as_index=False).agg(
        n=("question_id", "count"),
        valid_rate=("valid_answer", "mean"),
        avg_generation_docs=("generation_doc_count", "mean"),
        avg_context_tokens=("context_estimated_tokens", "mean"),
        median_context_tokens=("context_estimated_tokens", "median"),
        avg_context_chars=("context_chars", "mean"),
        avg_rouge_l=("rouge_l_f1", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
    )
    by_kb = details.groupby(["budget", "kb"], as_index=False).agg(
        n=("question_id", "count"),
        valid_rate=("valid_answer", "mean"),
        avg_generation_docs=("generation_doc_count", "mean"),
        avg_context_tokens=("context_estimated_tokens", "mean"),
        avg_rouge_l=("rouge_l_f1", "mean"),
        avg_latency_seconds=("latency_seconds", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    by_kb.to_csv(output_dir / "summary_by_kb.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    show["valid_rate"] = (show["valid_rate"]*100).round(2)
    show["avg_rouge_l"] = show["avg_rouge_l"].round(4)
    show["avg_context_tokens"] = show["avg_context_tokens"].round(1)
    show["avg_generation_docs"] = show["avg_generation_docs"].round(2)
    show["avg_latency_seconds"] = show["avg_latency_seconds"].round(3)
    print("\n" + "="*80)
    print("TÓM TẮT CONTEXT BUDGET ABLATION")
    print("="*80)
    print(show.to_string(index=False))
    print(f"\n✅ {output_dir / 'details.csv'}")
    print(f"✅ {output_dir / 'summary.csv'}")
    print(f"✅ {output_dir / 'summary_by_kb.csv'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-kb", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000,1500,2200,3000])
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output-dir", default="context_budget_results")
    args = parser.parse_args()
    budgets = sorted(set(x for x in args.budgets if x > 0))
    print(f"🔬 Budgets: {budgets}; {args.per_kb} câu/KB; seed={args.seed}")
    print("Lưu ý: context token là token ƯỚC LƯỢNG theo llm_chain.py.\n")
    questions = load_balanced_questions(args.per_kb, args.seed)
    run_experiment(questions, budgets, args.sleep, Path(args.output_dir))

if __name__ == "__main__":
    main()
