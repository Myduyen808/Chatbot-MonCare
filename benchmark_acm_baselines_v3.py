"""
Benchmark 5 chiến lược quản lý ngữ cảnh cho MomCare — phiên bản 2
=================================================================

Điểm khác so với v1:
- Kịch bản được xây dựng từ các câu hỏi và đáp án thật trong
  KB1_Medical_Standard.xlsx.
- Câu hỏi nối tiếp bớt mơ hồ hơn nhưng vẫn cần lịch sử để khôi phục đầy đủ
  chủ đề, đối tượng hoặc độ tuổi.
- Mỗi kịch bản ghi rõ số thứ tự câu trong KB1 để đối chiếu.
- Chỉ đánh giá bước quản lý lịch sử + Query Rewriting nhằm cô lập ảnh hưởng
  của từng chiến lược memory.

Chạy thử một kịch bản:
    python benchmark_acm_baselines_v2.py --scenario vitamin_d

Chạy cả ba kịch bản:
    python benchmark_acm_baselines_v2.py --all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import time
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

load_dotenv()

from llm_chain import (
    rewrite_and_detect_intent,
    summarize_history_block,
    summarize_history_message,
)


# =============================================================================
# CẤU HÌNH ĐỒNG BỘ VỚI MOMCARE
# =============================================================================

FIXED_WINDOW_MESSAGES = 6
ACM_HISTORY_MAX_MESSAGES = 20
HISTORY_KEEP_RECENT = 2
HISTORY_LIGHT_THRESHOLD = 1000
HISTORY_STRONG_THRESHOLD = 2500
HISTORY_MESSAGE_SUMMARY_MIN_CHARS = 250

SLEEP_BETWEEN_METHODS = 3
OUTPUT_PREFIX = "acm_baseline_comparison_v2"

METHODS = [
    "no_memory",
    "fixed_window",
    "full_history",
    "summary_only",
    "acm",
]


# =============================================================================
# KIỂU DỮ LIỆU
# =============================================================================

@dataclass
class Scenario:
    scenario_id: str
    title: str
    history: list[BaseMessage]
    question: str
    required_groups: list[list[str]]
    source_note: str


# =============================================================================
# HÀM HỖ TRỢ
# =============================================================================

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def count_chars(history: list[BaseMessage]) -> int:
    return sum(len(str(message.content)) for message in history)


def to_dicts(history: list[BaseMessage]) -> list[dict[str, str]]:
    return [
        {
            "type": (
                "human"
                if isinstance(message, HumanMessage)
                else "ai"
            ),
            "content": str(message.content),
        }
        for message in history
    ]


def parse_tokens(log_text: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(
            r"Tốn:\s*(\d+)\s*tokens",
            log_text,
        )
    ]


def check_rewrite(
    rewritten_question: str,
    required_groups: list[list[str]],
) -> tuple[bool, list[str], list[list[str]]]:
    """
    Mỗi nhóm phải khớp ít nhất một cách diễn đạt.

    Ví dụ:
    [
        ["vitamin d"],
        ["6 tháng", "trẻ 6 tháng"],
        ["nguồn bổ sung", "bổ sung từ"]
    ]
    """
    normalized = normalize(rewritten_question)
    matched: list[str] = []
    missing: list[list[str]] = []

    for group in required_groups:
        found = next(
            (
                keyword
                for keyword in group
                if normalize(keyword) in normalized
            ),
            None,
        )

        if found is None:
            missing.append(group)
        else:
            matched.append(found)

    return not missing, matched, missing


# =============================================================================
# CÁC CHIẾN LƯỢC QUẢN LÝ LỊCH SỬ
# =============================================================================

def summary_only(
    history: list[BaseMessage],
) -> tuple[list[BaseMessage], str]:
    if not history:
        return [], "summary_only_empty"

    summary = summarize_history_block(to_dicts(history))

    if summary:
        return [
            AIMessage(
                content="[Tóm tắt toàn bộ lịch sử] " + summary
            )
        ], "summary_only"

    return list(history), "summary_only_fallback_full"


def build_acm(
    history: list[BaseMessage],
) -> tuple[list[BaseMessage], str]:
    """
    Giữ đúng logic ACM đang dùng trong MomCare:
    - ACM xét tối đa 20 tin nhắn gần nhất;
    - dưới 1000 ký tự: keep_all;
    - 1000–2499 ký tự: light_summary;
    - từ 2500 ký tự: strong_summary;
    - giữ nguyên 2 tin nhắn gần nhất khi tóm tắt.
    """
    recent = list(history[-ACM_HISTORY_MAX_MESSAGES:])

    if not recent:
        return [], "empty"

    total = count_chars(recent)

    if total < HISTORY_LIGHT_THRESHOLD:
        return recent, "keep_all"

    split_index = max(
        0,
        len(recent) - HISTORY_KEEP_RECENT,
    )
    old_messages = recent[:split_index]
    recent_tail = recent[split_index:]

    if total < HISTORY_STRONG_THRESHOLD:
        processed: list[BaseMessage] = []

        for message in old_messages:
            content = str(message.content)

            if (
                len(content)
                >= HISTORY_MESSAGE_SUMMARY_MIN_CHARS
            ):
                summarized = summarize_history_message(content)
                if summarized:
                    content = summarized

            if isinstance(message, HumanMessage):
                processed.append(
                    HumanMessage(content=content)
                )
            else:
                processed.append(
                    AIMessage(content=content)
                )

        processed.extend(recent_tail)
        return processed, "light_summary"

    summary = summarize_history_block(
        to_dicts(old_messages)
    )

    if summary:
        return [
            AIMessage(
                content="[Tóm tắt lịch sử cũ] " + summary
            ),
            *recent_tail,
        ], "strong_summary"

    return recent, "strong_summary_fallback_full"


def prepare_history(
    method: str,
    history: list[BaseMessage],
) -> tuple[list[BaseMessage], str]:
    if method == "no_memory":
        return [], "none"

    if method == "fixed_window":
        return (
            list(history[-FIXED_WINDOW_MESSAGES:]),
            f"last_{FIXED_WINDOW_MESSAGES}_messages",
        )

    if method == "full_history":
        return list(history), "full"

    if method == "summary_only":
        return summary_only(history)

    if method == "acm":
        return build_acm(history)

    raise ValueError(f"Phương pháp không hợp lệ: {method}")


# =============================================================================
# DỮ LIỆU THẬT TỪ KB1
# =============================================================================

# Các câu dưới đây lấy từ:
# KB1_Medical_Standard.xlsx
#
# Ăn dặm:
# - câu 19: bắt đầu ăn bổ sung khi tròn 6 tháng;
# - câu 20: từ loãng đến đặc, từ ít đến nhiều;
# - câu 289–293: thời điểm, tác hại ăn sớm, độ đặc và nhóm chất.
#
# Răng miệng:
# - câu 196: mọc răng, đau và quấy khóc;
# - câu 198: thời điểm và dụng cụ vệ sinh răng miệng;
# - câu 199–200: khám định kỳ và khả năng khám phá.
#
# Vitamin D:
# - câu 398: lý do trẻ bú mẹ cần bổ sung vitamin D;
# - câu 399: hậu quả thiếu vitamin D;
# - câu 400: các nguồn bổ sung vitamin D.


SCENARIOS: dict[str, Scenario] = {
    "vitamin_d": Scenario(
        scenario_id="vitamin_d",
        title="Vitamin D ở trẻ bú mẹ trong 6 tháng đầu",
        history=[
            HumanMessage(content=(
                "Bé nhà tôi hiện được 6 tháng tuổi và vẫn bú mẹ là chính. "
                "Vì sao trẻ bú mẹ hoàn toàn trong 6 tháng đầu cần bổ sung vitamin D?"
            )),
            AIMessage(content=(
                "Hàm lượng vitamin D trong sữa mẹ thấp và có thể không đáp ứng đủ "
                "nhu cầu phát triển của trẻ. Chủ đề đang trao đổi là vitamin D "
                "cho trẻ 6 tháng tuổi bú mẹ."
            )),
            HumanMessage(content="Thiếu vitamin D có thể ảnh hưởng như thế nào?"),
            AIMessage(content=(
                "Thiếu vitamin D có thể liên quan đến còi xương, biến dạng xương "
                "hoặc chậm mọc răng."
            )),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content=(
                "Tôi muốn hỏi tiếp về nguồn cung cấp của loại vitamin đã nói ở đầu "
                "cuộc trò chuyện, nhưng không nhắc lại tên."
            )),
            AIMessage(content=(
                "MomCare sẽ dựa vào toàn bộ ngữ cảnh đã được quản lý để xác định "
                "loại vitamin và đối tượng đang được nhắc tới."
            )),
        ],
        question="Loại vitamin đó còn có thể được bổ sung từ những nguồn nào?",
        required_groups=[
            ["vitamin d"],
            ["6 tháng", "trẻ 6 tháng"],
            ["nguồn bổ sung", "bổ sung từ", "nguồn nào", "nguồn thực phẩm"],
        ],
        source_note="KB1 câu 398–400.",
    ),

    "oral_care": Scenario(
        scenario_id="oral_care",
        title="Vệ sinh răng miệng khi trẻ bắt đầu mọc răng",
        history=[
            HumanMessage(content=(
                "Bé nhà tôi hiện được 6 tháng tuổi và chưa mọc răng rõ. "
                "Em nên vệ sinh răng miệng cho bé từ khi nào và dùng dụng cụ gì?"
            )),
            AIMessage(content=(
                "Nên vệ sinh từ khi bé chưa mọc răng bằng gạc mềm thấm nước sạch; "
                "khi mọc răng thì dùng bàn chải lông mềm dành cho trẻ nhỏ."
            )),
            HumanMessage(content=(
                "Bé thường đưa tay vào miệng, chảy nước dãi và thích cắn đồ vật."
            )),
            AIMessage(content=(
                "Các biểu hiện này có thể gặp trong giai đoạn phát triển hoặc "
                "chuẩn bị mọc răng."
            )),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content=(
                "Tôi muốn hỏi lại phần dụng cụ chăm sóc vùng miệng đã nói ở đầu "
                "cuộc trò chuyện."
            )),
            AIMessage(content=(
                "MomCare sẽ dùng ngữ cảnh trước đó để xác định đúng độ tuổi và "
                "chủ đề chăm sóc đang được hỏi."
            )),
        ],
        question="Còn khi bé chưa mọc răng thì nên dùng dụng cụ gì để vệ sinh?",
        required_groups=[
            ["vệ sinh răng miệng", "vệ sinh miệng", "làm sạch lợi"],
            ["6 tháng", "trẻ 6 tháng"],
            ["gạc mềm", "khăn mềm", "dụng cụ"],
        ],
        source_note="KB1 câu 196 và 198.",
    ),

    "complementary_feeding": Scenario(
        scenario_id="complementary_feeding",
        title="Độ đặc và lượng thức ăn khi trẻ bắt đầu ăn dặm",
        history=[
            HumanMessage(content=(
                "Bé nhà tôi vừa tròn 6 tháng tuổi và đang bắt đầu ăn dặm. "
                "Nguyên tắc ban đầu là cho ăn từ loãng đến đặc, từ ít đến nhiều."
            )),
            AIMessage(content=(
                "Trẻ tròn 6 tháng tuổi có thể bắt đầu ăn bổ sung và vẫn tiếp tục "
                "bú mẹ; lượng và độ đặc cần tăng dần theo khả năng thích nghi."
            )),
            HumanMessage(content=(
                "Nếu cho trẻ ăn dặm quá sớm thì có thể ảnh hưởng hệ tiêu hóa "
                "và làm trẻ bú mẹ ít hơn."
            )),
            AIMessage(content=(
                "Đúng, cần theo dõi khả năng dung nạp và không thay thế sữa mẹ "
                "quá sớm."
            )),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            AIMessage(content="Gia đình tiếp tục theo dõi tình trạng bú, ngủ, tiêu hóa và sinh hoạt hằng ngày. Bé hiện không sốt, không nôn, không tiêu chảy và vẫn chơi bình thường. Thông tin này được ghi lại để trao đổi với nhân viên y tế khi cần thiết."),
            HumanMessage(content=(
                "Tôi muốn hỏi tiếp nguyên tắc về kết cấu và lượng thức ăn đã nói "
                "ở đầu cuộc trò chuyện."
            )),
            AIMessage(content=(
                "MomCare sẽ dựa vào ngữ cảnh đã quản lý để khôi phục đúng độ tuổi "
                "và chủ đề ăn bổ sung."
            )),
        ],
        question="Còn độ đặc và số lượng thì áp dụng nguyên tắc nào cho bé?",
        required_groups=[
            ["ăn bổ sung", "ăn dặm"],
            ["6 tháng", "trẻ 6 tháng"],
            ["độ đặc", "loãng", "số lượng", "lượng thức ăn"],
        ],
        source_note="KB1 câu 19–20 và 289–292.",
    ),
}


# =============================================================================
# CHẠY THỰC NGHIỆM
# =============================================================================

def run_method(
    scenario: Scenario,
    method: str,
) -> dict:
    raw_chars = count_chars(scenario.history)
    raw_messages = len(scenario.history)

    captured = io.StringIO()
    start = time.perf_counter()

    processed: list[BaseMessage] = []
    mode = "error"
    rewritten = ""
    intent = ""
    error = ""

    try:
        with contextlib.redirect_stdout(captured):
            processed, mode = prepare_history(
                method,
                scenario.history,
            )
            rewritten, intent = rewrite_and_detect_intent(
                scenario.question,
                processed,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start
    log_text = captured.getvalue()

    if log_text.strip():
        print(log_text.rstrip())

    tokens = parse_tokens(log_text)

    context_ok, matched, missing = check_rewrite(
        rewritten,
        scenario.required_groups,
    )

    processed_chars = count_chars(processed)
    reduction = (
        (raw_chars - processed_chars)
        / raw_chars
        * 100
        if raw_chars
        else 0.0
    )

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_title": scenario.title,
        "source_note": scenario.source_note,
        "method": method,
        "memory_mode": mode,
        "question": scenario.question,
        "raw_history_messages": raw_messages,
        "processed_history_messages": len(processed),
        "raw_history_chars": raw_chars,
        "processed_history_chars": processed_chars,
        "history_reduction_percent": round(
            reduction,
            2,
        ),
        "api_calls_logged": len(tokens),
        "api_tokens_logged_total": sum(tokens),
        "api_tokens_logged_each": " | ".join(
            map(str, tokens)
        ),
        "elapsed_s": round(elapsed, 3),
        "rewritten_question": rewritten,
        "intent": intent,
        "context_ok": context_ok,
        "matched_keywords": " | ".join(matched),
        "missing_keyword_groups": " | ".join(
            "(" + " OR ".join(group) + ")"
            for group in missing
        ),
        "error": error,
    }


def run_scenario(
    scenario: Scenario,
) -> pd.DataFrame:
    print("\n" + "═" * 88)
    print(f"KỊCH BẢN: {scenario.title}")
    print(f"Nguồn: {scenario.source_note}")
    print(f"Câu hỏi nối tiếp: {scenario.question}")
    print(
        f"Lịch sử gốc: {len(scenario.history)} tin nhắn | "
        f"{count_chars(scenario.history)} ký tự"
    )
    print("═" * 88)

    rows: list[dict] = []

    for index, method in enumerate(METHODS, start=1):
        print("\n" + "─" * 88)
        print(
            f"[{index}/{len(METHODS)}] "
            f"PHƯƠNG PHÁP: {method}"
        )

        row = run_method(scenario, method)
        rows.append(row)

        print(
            f"History: "
            f"{row['processed_history_messages']} tin | "
            f"{row['processed_history_chars']} ký tự | "
            f"giảm "
            f"{row['history_reduction_percent']:.1f}%"
        )
        print(
            "Context: "
            + (
                "✅ Đạt"
                if row["context_ok"]
                else "❌ Không đạt"
            )
        )
        print(f"Rewrite: {row['rewritten_question']}")
        print(
            f"Khớp: {row['matched_keywords'] or '--'}"
        )

        if row["missing_keyword_groups"]:
            print(
                "Thiếu: "
                + row["missing_keyword_groups"]
            )

        print(
            f"API calls: {row['api_calls_logged']} | "
            f"tokens log: "
            f"{row['api_tokens_logged_total']} | "
            f"time: {row['elapsed_s']:.2f}s"
        )

        if row["error"]:
            print(f"Lỗi: {row['error']}")

        time.sleep(SLEEP_BETWEEN_METHODS)

    return pd.DataFrame(rows)


def summarize_results(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        detail
        .groupby("method", as_index=False)
        .agg(
            scenarios=("scenario_id", "count"),
            context_successes=("context_ok", "sum"),
            context_success_rate=("context_ok", "mean"),
            avg_processed_messages=(
                "processed_history_messages",
                "mean",
            ),
            avg_processed_chars=(
                "processed_history_chars",
                "mean",
            ),
            avg_history_reduction_percent=(
                "history_reduction_percent",
                "mean",
            ),
            avg_api_calls=("api_calls_logged", "mean"),
            avg_logged_tokens=(
                "api_tokens_logged_total",
                "mean",
            ),
            avg_elapsed_s=("elapsed_s", "mean"),
        )
    )

    summary["context_success_rate"] = (
        summary["context_success_rate"] * 100
    ).round(2)

    numeric_columns = [
        "avg_processed_messages",
        "avg_processed_chars",
        "avg_history_reduction_percent",
        "avg_api_calls",
        "avg_logged_tokens",
        "avg_elapsed_s",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns].round(2)
    )

    return summary


def save_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    detail.to_csv(
        f"{OUTPUT_PREFIX}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    detail.to_excel(
        f"{OUTPUT_PREFIX}_detail.xlsx",
        index=False,
    )
    summary.to_csv(
        f"{OUTPUT_PREFIX}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_excel(
        f"{OUTPUT_PREFIX}_summary.xlsx",
        index=False,
    )

    print(
        "\n✅ Đã lưu 4 file CSV/XLSX với tiền tố "
        f"{OUTPUT_PREFIX}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark chiến lược quản lý ngữ cảnh "
            "trên câu hỏi thật từ KB1."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default="vitamin_d",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Chạy cả ba kịch bản.",
    )
    args = parser.parse_args()

    selected = (
        list(SCENARIOS.values())
        if args.all
        else [SCENARIOS[args.scenario]]
    )

    detail = pd.concat(
        [
            run_scenario(scenario)
            for scenario in selected
        ],
        ignore_index=True,
    )

    summary = summarize_results(detail)

    save_outputs(detail, summary)

    print("\n" + "═" * 88)
    print("TỔNG KẾT")
    print("═" * 88)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
