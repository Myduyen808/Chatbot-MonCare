from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import time
import json
import os
import re

from llm_chain import RAGChain
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ==========================================================
# KHỞI TẠO
# ==========================================================
TOP_K = 5
chain = RAGChain(k=TOP_K)

smoother = SmoothingFunction().method1
rouge_scorer_ = rouge_scorer.RougeScorer(
    ['rougeL'],
    use_stemmer=False
)

# ==========================================================
# GỌI CHAIN + AUTO RETRY RATE LIMIT
# ==========================================================
def call_chain_with_retry(q, max_retries=5):

    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "question": q,
                "history": []
            })
            return result

        except Exception as e:
            err = str(e)

            # Rate limit Groq
            if "429" in err or "rate_limit" in err.lower():

                wait = 60

                try:
                    # Parse kiểu:
                    # "Please try again in 2m34.5s"
                    m = re.search(
                        r'in (\d+)m([\d.]+)s',
                        err
                    )

                    if m:
                        wait = (
                            int(m.group(1)) * 60
                            + float(m.group(2))
                            + 5
                        )

                except Exception:
                    pass

                print(
                    f"⚠️ Rate limit → "
                    f"chờ {wait:.0f}s "
                    f"(lần {attempt+1}/{max_retries})"
                )

                time.sleep(wait)

            else:
                print(f"❌ Lỗi invoke: {e}")
                return None

    print("❌ Retry thất bại.")
    return None


# ==========================================================
# BLEU + ROUGE
# ==========================================================
def compute_metrics(answers, references):

    bleu_scores = []
    rouge_scores = []

    for answer, reference in zip(
        answers,
        references
    ):

        answer = str(answer).strip()
        reference = str(reference).strip()

        if not answer or not reference:
            continue

        try:
            bleu = sentence_bleu(
                [reference.split()],
                answer.split(),
                smoothing_function=smoother
            )

            rouge = rouge_scorer_.score(
                reference,
                answer
            )['rougeL'].fmeasure

            bleu_scores.append(bleu)
            rouge_scores.append(rouge)

        except Exception as e:
            print(f"⚠️ Skip metric lỗi: {e}")

    avg_bleu = (
        sum(bleu_scores) / len(bleu_scores)
        if bleu_scores else 0
    )

    avg_rouge = (
        sum(rouge_scores) / len(rouge_scores)
        if rouge_scores else 0
    )

    return avg_bleu, avg_rouge


# ==========================================================
# RUN SCENARIO
# ==========================================================
def run_scenario(
    file_path,
    scenario_name,
    sample=400
):

    # ---------------------------------
    # CHECK FILE
    # ---------------------------------
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return None

    # ---------------------------------
    # CHECKPOINT
    # ---------------------------------
    safe_name = (
        scenario_name
        .replace(" ", "_")
        .replace("-", "_")
    )

    checkpoint_file = (
        f"checkpoint_{safe_name}.json"
    )

    output_excel = (
        f"result_{safe_name}.xlsx"
    )

    questions = []
    answers = []
    references = []
    start_idx = 0

    # Resume nếu có checkpoint
    if os.path.exists(checkpoint_file):

        try:
            with open(
                checkpoint_file,
                "r",
                encoding="utf-8"
            ) as f:

                ck = json.load(f)

            questions = ck.get(
                "questions", []
            )

            answers = ck.get(
                "answers", []
            )

            references = ck.get(
                "references", []
            )

            start_idx = len(questions)

            print(
                f"🔄 Resume checkpoint "
                f"({start_idx} câu)"
            )

        except Exception as e:
            print(
                f"⚠️ Lỗi đọc checkpoint: {e}"
            )

    # ---------------------------------
    # LOAD DATA
    # ---------------------------------
    df = pd.read_excel(file_path)

    df = df.head(sample)

    print("\n" + "=" * 70)
    print(
        f"KỊCH BẢN: {scenario_name}"
    )
    print(f"TOP-K = {TOP_K}")
    print(
        f"Số câu: {len(df)}"
    )
    print(
        f"Bắt đầu từ câu: "
        f"{start_idx + 1}"
    )
    print("=" * 70)

    # ---------------------------------
    # LOOP
    # ---------------------------------
    for idx in range(
        start_idx,
        len(df)
    ):

        row = df.iloc[idx]

        q = str(
            row[
                'Câu hỏi người dùng (Input)'
            ]
        )

        ref = str(
            row[
                'Phản hồi kỳ vọng (Expected Output)'
            ]
        )

        print(
            f"\n[{idx+1}/{len(df)}]"
        )
        print(
            f"Q: {q[:80]}"
        )

        result = (
            call_chain_with_retry(q)
        )

        answer = (
            result.get("answer", "")
            if result
            else "Lỗi xử lý"
        )

        print(
            f"A: {answer[:100]}..."
        )

        questions.append(q)
        answers.append(answer)
        references.append(ref)

        # Delay tránh spam API
        time.sleep(3)

        # SAVE MỖI 20 CÂU
        if (idx + 1) % 20 == 0:

            ck_data = {
                "questions":
                    questions,
                "answers":
                    answers,
                "references":
                    references
            }

            with open(
                checkpoint_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    ck_data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"💾 Saved "
                f"{idx+1} câu"
            )

    # ======================================================
    # SAVE FINAL
    # ======================================================
    result_df = pd.DataFrame({
        "question":
            questions,
        "reference":
            references,
        "generated_answer":
            answers
    })

    result_df.to_excel(
        output_excel,
        index=False
    )

    print(
        f"\n✅ Saved: "
        f"{output_excel}"
    )

    # ======================================================
    # METRICS
    # ======================================================
    print(
        "\n📊 Đang tính BLEU + ROUGE-L..."
    )

    avg_bleu, avg_rouge = (
        compute_metrics(
            answers,
            references
        )
    )

    print("\n" + "=" * 70)
    print(
        f"KẾT QUẢ: "
        f"{scenario_name}"
    )
    print("=" * 70)

    print(
        f"Số mẫu   : "
        f"{len(questions)}"
    )

    print(
        f"BLEU      : "
        f"{avg_bleu:.4f}"
    )

    print(
        f"ROUGE-L   : "
        f"{avg_rouge:.4f}"
    )

    print("=" * 70)

    return {
        "bleu_score":
            avg_bleu,
        "rouge_score":
            avg_rouge
    }


# ==========================================================
# CHẠY KB1
# ==========================================================
r1 = run_scenario(
    file_path="KB3_Information_Noise.xlsx",
    scenario_name="KB3_Thong_Tin_Gay_Nhieu",
    sample=400
)

# ==========================================================
# SUMMARY
# ==========================================================
if r1:
    print("\n" + "=" * 70)
    print("TỔNG KẾT CUỐI")
    print("=" * 70)

    print(
        f"BLEU      : "
        f"{r1['bleu_score']:.4f}"
    )

    print(
        f"ROUGE-L   : "
        f"{r1['rouge_score']:.4f}"
    )