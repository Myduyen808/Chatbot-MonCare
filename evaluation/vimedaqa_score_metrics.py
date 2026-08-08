#!/usr/bin/env python3
"""Offline scoring for the leakage-controlled ViMedAQA benchmark.

Reads the checkpoint produced by ``vimedaqa_clean_benchmark.py generate`` and
computes generation metrics without calling Groq or any other LLM API.

Reported scores follow the 0--100 presentation used in the ViMedAQA paper:
    BERT = mean BERTScore F1
    BLEU = corpus SacreBLEU
    MET  = mean sentence METEOR
    ROU  = mean sentence ROUGE-L F1
    Avg  = arithmetic mean of the four metrics

The ViMedAQA paper names these metrics but does not publish the exact scoring
implementation/model configuration.  This script therefore records its own
metric configuration in ``generation_metrics_protocol.json`` so the thesis
results are reproducible and are not presented as an exact reproduction of the
paper's evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_VALID_SAMPLES = 2213
PAPER_URL = "https://aclanthology.org/2024.acl-srw.31/"
BERT_MODEL = "bert-base-multilingual-cased"
BLEU_TOKENIZER = "13a"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ViMedAQA generation_results.csv offline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/vimedaqa_clean/generation_results.csv"),
        help="Generation checkpoint CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/vimedaqa_clean"),
        help="Directory for metric outputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for BERTScore. Use cpu on low-VRAM machines.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="BERTScore batch size. Default 4 is conservative for 8 GB RAM.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow scoring fewer than 2213 unique OK samples.",
    )
    return parser.parse_args()


def require_dependencies() -> None:
    missing: list[str] = []
    checks = {
        "sacrebleu": "sacrebleu",
        "nltk": "nltk",
        "bert_score": "bert-score",
        "torch": "torch",
    }
    for module_name, pip_name in checks.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        packages = " ".join(dict.fromkeys(missing))
        raise SystemExit(
            "Thiếu thư viện chấm điểm. Trong mom_env hãy chạy:\n\n"
            f"    pip install {packages}\n\n"
            "Sau khi cài xong, chạy lại file này. Bước chấm điểm không gọi Groq."
        )


def load_unique_ok_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    if not path.exists():
        raise SystemExit(f"Không tìm thấy file input: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "row_index",
        "sample_id",
        "topic",
        "reference_answer",
        "prediction",
        "status",
    }
    actual = set(rows[0].keys()) if rows else set()
    missing = sorted(required - actual)
    if missing:
        raise SystemExit(f"generation_results.csv thiếu cột: {', '.join(missing)}")

    historical_errors = sum(1 for row in rows if row.get("status") == "ERROR")

    # Keep the last successful record for each row_index.  Historical ERROR
    # rows remain in the checkpoint for auditability but never enter scoring.
    unique: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        try:
            idx = int(row["row_index"])
        except (TypeError, ValueError):
            raise SystemExit(f"row_index không hợp lệ: {row.get('row_index')!r}")
        if not row.get("reference_answer", "").strip():
            raise SystemExit(f"Reference rỗng tại row_index={idx}")
        if not row.get("prediction", "").strip():
            raise SystemExit(f"Prediction rỗng tại row_index={idx}")
        unique[idx] = row

    ordered = [unique[idx] for idx in sorted(unique)]
    return ordered, historical_errors


def unicode_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().strip())


def ensure_nltk_resources() -> None:
    """Prepare resources used by NLTK METEOR's standard WordNet matching."""
    import nltk

    for resource, download_name in (
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            print(f"Downloading NLTK resource: {download_name}")
            nltk.download(download_name, quiet=False)


def compute_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    """Unicode-aware ROUGE-L F1, returned per sample on a 0--1 scale."""

    def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
        # O(min(m,n)) memory dynamic program; answer strings are short.
        if len(a) < len(b):
            short, long_ = a, b
        else:
            short, long_ = b, a
        dp = [0] * (len(short) + 1)
        for token_long in long_:
            prev = 0
            for j, token_short in enumerate(short, start=1):
                old = dp[j]
                if token_long == token_short:
                    dp[j] = prev + 1
                elif dp[j - 1] > dp[j]:
                    dp[j] = dp[j - 1]
                prev = old
        return dp[-1]

    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        p = unicode_tokens(pred)
        r = unicode_tokens(ref)
        if not p or not r:
            scores.append(0.0)
            continue
        lcs = lcs_len(p, r)
        precision = lcs / len(p)
        recall = lcs / len(r)
        denom = precision + recall
        scores.append(0.0 if denom == 0 else 2 * precision * recall / denom)
    return scores


def compute_meteor(predictions: Sequence[str], references: Sequence[str]) -> list[float]:
    from nltk.translate.meteor_score import meteor_score

    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        pred_tokens = unicode_tokens(pred)
        ref_tokens = unicode_tokens(ref)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
        else:
            scores.append(float(meteor_score([ref_tokens], pred_tokens)))
    return scores


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    device: str,
    batch_size: int,
) -> list[float]:
    from bert_score import score as bert_score

    print(
        f"BERTScore: model={BERT_MODEL} | device={device} | "
        f"batch_size={batch_size}"
    )
    _, _, f1 = bert_score(
        list(predictions),
        list(references),
        model_type=BERT_MODEL,
        lang="vi",
        batch_size=batch_size,
        device=device,
        verbose=True,
        rescale_with_baseline=False,
    )
    return [float(x) for x in f1.cpu().tolist()]


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    import sacrebleu

    return float(
        sacrebleu.corpus_bleu(
            list(predictions),
            [list(references)],
            tokenize=BLEU_TOKENIZER,
        ).score
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bool_value(value: str) -> bool | None:
    value = (value or "").strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def score_group(
    name: str,
    indices: Sequence[int],
    predictions: Sequence[str],
    references: Sequence[str],
    rouge_l: Sequence[float],
    meteor: Sequence[float],
    bert_f1: Sequence[float],
) -> dict[str, object]:
    preds = [predictions[i] for i in indices]
    refs = [references[i] for i in indices]
    bert = mean(bert_f1[i] for i in indices) * 100
    bleu = corpus_bleu(preds, refs)
    met = mean(meteor[i] for i in indices) * 100
    rou = mean(rouge_l[i] for i in indices) * 100
    avg = (bert + bleu + met + rou) / 4
    return {
        "group": name,
        "n": len(indices),
        "BERT": round(bert, 4),
        "BLEU": round(bleu, 4),
        "MET": round(met, 4),
        "ROU": round(rou, 4),
        "Avg": round(avg, 4),
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size phải >= 1")

    require_dependencies()
    rows, historical_errors = load_unique_ok_rows(args.input)

    if len(rows) != EXPECTED_VALID_SAMPLES and not args.allow_incomplete:
        raise SystemExit(
            f"Có {len(rows)} mẫu OK duy nhất, cần {EXPECTED_VALID_SAMPLES}. "
            "Nếu cố ý chấm tập chưa hoàn chỉnh, thêm --allow-incomplete."
        )

    print("\n========== VIMEDAQA OFFLINE SCORING ==========")
    print(f"Unique OK samples : {len(rows)}")
    print(f"Historical errors : {historical_errors} (excluded from scoring)")
    print(f"Input             : {args.input}")
    print("Groq/API calls    : 0")
    print("==============================================\n")

    predictions = [row["prediction"].strip() for row in rows]
    references = [row["reference_answer"].strip() for row in rows]

    print("[1/3] Computing Unicode ROUGE-L...")
    rouge_l = compute_rouge_l(predictions, references)

    print("[2/3] Computing METEOR...")
    ensure_nltk_resources()
    meteor = compute_meteor(predictions, references)

    print("[3/3] Computing BERTScore (this is the slow step)...")
    bert_f1 = compute_bertscore(
        predictions,
        references,
        device=args.device,
        batch_size=args.batch_size,
    )

    all_indices = list(range(len(rows)))
    groups: list[tuple[str, list[int]]] = [("ALL", all_indices)]

    topics = sorted({row["topic"] for row in rows})
    for topic in topics:
        indices = [i for i, row in enumerate(rows) if row["topic"] == topic]
        groups.append((f"topic:{topic}", indices))

    if "gold_context_hit5" in rows[0]:
        hit = [
            i for i, row in enumerate(rows)
            if bool_value(row.get("gold_context_hit5", "")) is True
        ]
        miss = [
            i for i, row in enumerate(rows)
            if bool_value(row.get("gold_context_hit5", "")) is False
        ]
        if hit:
            groups.append(("retrieval:hit@5", hit))
        if miss:
            groups.append(("retrieval:miss@5", miss))

    summary = [
        score_group(
            name,
            indices,
            predictions,
            references,
            rouge_l,
            meteor,
            bert_f1,
        )
        for name, indices in groups
    ]

    per_sample: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        per_sample.append(
            {
                "row_index": row["row_index"],
                "sample_id": row["sample_id"],
                "topic": row["topic"],
                "gold_context_hit5": row.get("gold_context_hit5", ""),
                "ROUGE_L_F1": round(rouge_l[i] * 100, 4),
                "METEOR": round(meteor[i] * 100, 4),
                "BERTScore_F1": round(bert_f1[i] * 100, 4),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "generation_metrics_summary.csv"
    sample_path = args.out_dir / "generation_metrics_per_sample.csv"
    protocol_path = args.out_dir / "generation_metrics_protocol.json"

    write_csv(summary_path, summary)
    write_csv(sample_path, per_sample)

    protocol = {
        "benchmark": "ViMedAQA clean context-only RAG benchmark",
        "paper": PAPER_URL,
        "input": str(args.input),
        "unique_ok_samples": len(rows),
        "historical_error_rows_excluded": historical_errors,
        "score_scale": "0-100",
        "BLEU": {
            "implementation": "sacrebleu.corpus_bleu",
            "tokenizer": BLEU_TOKENIZER,
            "aggregation": "corpus",
        },
        "ROUGE_L": {
            "implementation": "Unicode token LCS",
            "tokenization": r"lowercase regex: \w+|[^\w\s]",
            "aggregation": "mean sentence F1",
        },
        "METEOR": {
            "implementation": "nltk.translate.meteor_score.meteor_score",
            "tokenization": r"lowercase regex: \w+|[^\w\s]",
            "aggregation": "mean sentence score",
        },
        "BERTScore": {
            "implementation": "bert-score",
            "component": "F1",
            "model": BERT_MODEL,
            "language": "vi",
            "rescale_with_baseline": False,
            "device": args.device,
            "batch_size": args.batch_size,
            "aggregation": "mean sentence F1",
        },
        "Avg": "arithmetic mean of BERT, BLEU, MET, ROU",
        "comparability_note": (
            "The ViMedAQA paper names BLEU, METEOR, ROUGE-L and BERTScore "
            "but does not publish the exact evaluation implementation/model "
            "configuration. These results use the explicit protocol above."
        ),
    }
    with protocol_path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    print("\n========== VIMEDAQA GENERATION METRICS ==========")
    print(f"{'Group':<24} {'N':>5} {'BERT':>8} {'BLEU':>8} {'MET':>8} {'ROU':>8} {'Avg':>8}")
    print("-" * 76)
    for row in summary:
        print(
            f"{str(row['group']):<24} {int(row['n']):>5} "
            f"{float(row['BERT']):>8.2f} {float(row['BLEU']):>8.2f} "
            f"{float(row['MET']):>8.2f} {float(row['ROU']):>8.2f} "
            f"{float(row['Avg']):>8.2f}"
        )
    print("=================================================")
    print(f"Summary   : {summary_path}")
    print(f"Per-sample: {sample_path}")
    print(f"Protocol  : {protocol_path}")
    print("Groq calls: 0")


if __name__ == "__main__":
    main()
