"""Benchmark chi phi token cho ACM va Task Merging cua MomCare.

Muc tieu
--------
Do truc tiep ``prompt_tokens`` do Groq tra ve cho 4 cau hinh:

1. RAW_SEPARATE: gui toan bo lich su + Rewrite/Intent tach 2 lan goi.
2. RAW_MERGED:   gui toan bo lich su + Rewrite/Intent gop 1 lan goi.
3. ACM_SEPARATE: Rolling Summary + Rewrite/Intent tach 2 lan goi.
4. ACM_MERGED:   Rolling Summary + Task Merging (gan voi MomCare production).

Benchmark CHI do phan quan ly hoi thoai va xu ly truy van. Khong chay Retrieval,
Generation hay Guardrails, de khong tron chi phi cua cac tang khac vao bien dang
duoc khao sat.

Rolling Summary dung dung cac nguong production hien tai:
  - giu 2 tin nhan gan nhat;
  - chi cap nhat khi co >= 2 tin nhan cu chua tom tat;
  - phan cho tom tat phai co >= 250 ky tu.

Chi phi cua loi goi LLM de tao/cap nhat summary DUOC tinh vao tong token ACM.

Cach dung (dat file trong thu muc evaluation/ hoac thu muc goc project):

    python evaluation\\benchmark_token_efficiency.py

Hoac:

    python evaluation\\benchmark_token_efficiency.py --turns 10

Ket qua:
  evaluation/token_efficiency_results.csv
  evaluation/token_efficiency_summary.csv
  evaluation/token_efficiency_calls.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "llm_chain.py").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "llm_chain.py").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))

try:
    import llm_chain
    from langchain_core.messages import AIMessage, HumanMessage
except Exception as exc:  # pragma: no cover - chi xay ra khi thieu moi truong project
    raise SystemExit(
        "Khong import duoc llm_chain.py. Hay dat file benchmark trong thu muc "
        "evaluation/ cua project MomCare (hoac ngay thu muc goc project).\n"
        f"Chi tiet: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Cau hinh phai trung voi application.py production.
# ---------------------------------------------------------------------------
HISTORY_KEEP_RECENT = 2
ROLLING_SUMMARY_MIN_MESSAGES = 2
ROLLING_SUMMARY_TRIGGER_CHARS = 250


# ---------------------------------------------------------------------------
# Hoi thoai co dinh de moi cau hinh nhan DUNG CUNG MOT du lieu.
# Day la du lieu mo phong co kiem soat, khong phai log nguoi dung that.
# Noi dung chi dung de tao kich thuoc/ngữ canh hoi thoai cho benchmark token.
# ---------------------------------------------------------------------------
CONTROLLED_TURNS = [
    (
        "Bé nhà tôi 8 tháng tuổi, tôi muốn tìm hiểu về ăn dặm và vẫn tiếp tục "
        "cho bé bú mẹ. Tôi nên lưu ý những thông tin nào trong giai đoạn này?",
        "MomCare đã ghi nhận bé 8 tháng tuổi và chủ đề đang trao đổi là ăn dặm "
        "kết hợp với bú mẹ. Các câu hỏi tiếp theo sẽ được tra cứu theo đúng độ "
        "tuổi và nội dung mẹ đang quan tâm.",
    ),
    (
        "Nếu tôi hỏi tiếp về số bữa ăn dặm mỗi ngày thì hệ thống có giữ được "
        "thông tin bé đang 8 tháng tuổi từ câu trước không?",
        "Có. Trong lượt trao đổi hiện tại, thông tin bé 8 tháng tuổi là ngữ cảnh "
        "cần được giữ để làm rõ các câu hỏi rút gọn ở những lượt sau.",
    ),
    (
        "Còn sữa mẹ thì sao, tôi vẫn muốn hỏi cho chính bé 8 tháng tuổi ở trên "
        "chứ không chuyển sang một trẻ ở nhóm tuổi khác.",
        "MomCare ghi nhận câu hỏi vẫn nói về cùng bé 8 tháng tuổi và chủ đề hiện "
        "tại liên quan đến sữa mẹ. Hệ thống cần giữ đúng đối tượng khi viết lại "
        "truy vấn.",
    ),
    (
        "Nếu tôi chỉ viết ngắn là 'còn vitamin thì sao' thì thông tin tuổi của "
        "bé và chủ đề trước đó có được dùng để làm rõ câu hỏi không?",
        "Câu hỏi rút gọn cần được kết hợp với ngữ cảnh hội thoại còn phù hợp, "
        "đặc biệt là đối tượng và độ tuổi đã được người dùng nêu trước đó.",
    ),
    (
        "Tôi muốn quay lại chủ đề ăn dặm. Khi hỏi tiếp mà không nhắc lại tuổi, "
        "hãy hiểu rằng tôi vẫn đang nói về bé 8 tháng tuổi của mình.",
        "Ngữ cảnh chính vẫn là bé 8 tháng tuổi. Khi câu hỏi sau lược bỏ chủ thể, "
        "thông tin này có thể được dùng để tạo truy vấn độc lập phục vụ tìm kiếm.",
    ),
    (
        "Nếu câu tiếp theo của tôi là 'nên ăn mấy bữa' thì hệ thống cần viết lại "
        "thành câu đầy đủ nhưng không tự thêm những vấn đề tôi chưa hỏi.",
        "Đúng. Query Rewriting có nhiệm vụ bổ sung phần ngữ cảnh đang thiếu, "
        "không mở rộng sang mục đích hoặc nội dung mới ngoài câu hỏi hiện tại.",
    ),
    (
        "Sau nhiều lượt như vậy, tôi vẫn muốn hệ thống nhớ bé 8 tháng tuổi nhưng "
        "không cần gửi lại nguyên văn toàn bộ các câu trao đổi từ đầu.",
        "Rolling Summary được dùng để giữ thông tin cũ cần thiết dưới dạng ngắn "
        "hơn, trong khi hai tin nhắn gần nhất vẫn được giữ nguyên để hạn chế mất "
        "ngữ cảnh gần.",
    ),
    (
        "Bây giờ tôi hỏi 'còn sữa mẹ thì sao' một lần nữa; câu này vẫn thuộc "
        "chuỗi trao đổi về bé 8 tháng tuổi và không phải một chủ đề mới.",
        "Hệ thống cần nhận diện đây là câu hỏi tiếp nối, giữ đúng đối tượng và "
        "viết lại thành truy vấn độc lập trước khi chuyển sang bước truy xuất.",
    ),
    (
        "Nếu tôi hỏi thêm về ăn bổ sung thì vẫn giữ tuổi 8 tháng và đừng lấy "
        "khuyến nghị chỉ dành cho nhóm tuổi khác để áp vào bé.",
        "Độ tuổi là dữ kiện cần được giữ khi làm rõ truy vấn. Các bước grounding "
        "sau retrieval chịu trách nhiệm kiểm tra bằng chứng phù hợp với tuổi.",
    ),
    (
        "Ở lượt cuối này, tôi muốn biết hệ thống còn giữ được đối tượng đang hỏi "
        "là bé 8 tháng tuổi sau khi lịch sử đã dài hay không.",
        "Đây là lượt kiểm tra khả năng duy trì thông tin đối tượng khi lịch sử "
        "hội thoại dài lên và cơ chế Rolling Summary đã được kích hoạt.",
    ),
]


@dataclass
class MeterState:
    config: str = ""
    turn: int = 0
    phase: str = ""


METER = MeterState()
CALL_ROWS: list[dict[str, Any]] = []


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _measured_call_llm(
    prompt,
    system_prompt="Bạn là trợ lý MomCare, chuyên chăm sóc mẹ và bé.",
    temperature=None,
    max_retries=2,
    max_tokens=None,
    frequency_penalty=0.4,
    presence_penalty=0.3,
):
    """Ban tuong duong call_llm production, co ghi usage do Groq tra ve."""

    if temperature is None:
        temperature = getattr(llm_chain, "DEFAULT_TEMPERATURE", 0.0)

    keys = list(getattr(llm_chain, "_ALL_KEYS", []) or [])
    if not keys:
        raise RuntimeError("Khong tim thay GROQ_API_KEY trong cau hinh llm_chain.")

    # Benchmark uu tien lay du usage de so sanh token. Tang so retry so voi
    # production de mot dot TPM 429 ngan khong lam mat ca dong thuc nghiem.
    # Vi vay KHONG dung latency cua benchmark nay de ket luan hieu nang API.
    retries = max(4, int(max_retries))
    last_error: Exception | None = None

    for attempt in range(retries):
        started = time.perf_counter()
        try:
            client = llm_chain.Groq(
                api_key=random.choice(keys),
                timeout=20.0,
                max_retries=0,
            )
            kwargs = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "model": llm_chain.MODEL_NAME,
                "temperature": temperature,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - started
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            CALL_ROWS.append(
                {
                    "config": METER.config,
                    "turn": METER.turn,
                    "phase": METER.phase,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_chars": len(str(prompt)),
                    "latency_s": round(elapsed, 6),
                    "status": "OK",
                }
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - phu thuoc API
            last_error = exc
            if attempt + 1 < retries:
                error_text = str(exc)
                wait_s = 2.0
                if "429" in error_text:
                    # Groq thuong tra: "Please try again in 1.66s" hoac
                    # "in 2m3.5s". Chi cho benchmark cho toi da 30 giay;
                    # neu gap gioi han dai hon thi dung de nguoi dung chay lai sau.
                    match = re.search(
                        r"try again in\s+(?:(\d+)m)?([\d.]+)s",
                        error_text,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        minutes = int(match.group(1) or 0)
                        seconds = float(match.group(2) or 0.0)
                        wait_s = minutes * 60 + seconds + 0.75
                    if wait_s > 30:
                        raise RuntimeError(
                            "Groq dang cham gioi han dai hon 30 giay; "
                            "hay doi rate-limit reset roi chay lai config nay."
                        ) from exc
                time.sleep(wait_s)

    raise RuntimeError(f"Groq call that bai: {last_error}")


# Tat ca ham noi bo cua llm_chain (rewrite va rolling summary) se di qua meter.
ORIGINAL_CALL_LLM = llm_chain.call_llm
llm_chain.call_llm = _measured_call_llm


def _to_chat_messages(processed_messages: list[dict[str, str]]):
    result = []
    for message in processed_messages:
        content = str(message.get("content", ""))
        if message.get("type") == "human":
            result.append(HumanMessage(content=content))
        else:
            result.append(AIMessage(content=content))
    return result


def build_acm_history(
    safe_history: list[dict[str, str]],
    previous_summary: str,
    summarized_count: int,
):
    """Phan toi thieu cua build_adaptive_history trong application.py."""

    safe_history = list(safe_history or [])
    input_messages = len(safe_history)
    total_chars_before = sum(len(str(m.get("content", ""))) for m in safe_history)

    if not safe_history:
        return [], previous_summary, 0, False, total_chars_before, 0

    if summarized_count < 0 or summarized_count > input_messages:
        summarized_count = 0
        previous_summary = ""

    split_index = max(0, input_messages - HISTORY_KEEP_RECENT)
    if summarized_count > split_index:
        summarized_count = 0
        previous_summary = ""

    pending_messages = safe_history[summarized_count:split_index]
    pending_chars = sum(len(str(m.get("content", ""))) for m in pending_messages)
    enough_messages = len(pending_messages) >= ROLLING_SUMMARY_MIN_MESSAGES
    enough_chars = pending_chars >= ROLLING_SUMMARY_TRIGGER_CHARS

    updated_summary = previous_summary
    updated_count = summarized_count
    summary_updated = False

    if enough_messages and enough_chars:
        METER.phase = "summary_update"
        candidate = llm_chain.update_rolling_summary(
            previous_summary=previous_summary,
            new_messages=pending_messages,
        )
        if candidate:
            updated_summary = candidate
            updated_count = split_index
            summary_updated = True

    processed = []
    if updated_summary:
        processed.append(
            {
                "type": "ai",
                "content": "[Tóm tắt lịch sử cũ] " + updated_summary,
            }
        )
    processed.extend(safe_history[updated_count:])

    chat_messages = _to_chat_messages(processed)
    total_chars_after = sum(len(message.content) for message in chat_messages)
    return (
        chat_messages,
        updated_summary,
        updated_count,
        summary_updated,
        total_chars_before,
        total_chars_after,
    )


def build_raw_history(history: list[dict[str, str]]):
    messages = _to_chat_messages(history)
    chars = sum(len(message.content) for message in messages)
    return messages, chars


def _history_text(history) -> str:
    if not history:
        return "(không có lịch sử)"

    lines = []
    for msg in history:
        role = "Mẹ" if msg.__class__.__name__ == "HumanMessage" else "MomCare"
        content = re.sub(r"\s+", " ", str(msg.content)).strip()
        if len(content) > 500:
            content = content[:500].rsplit(" ", 1)[0] + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(không có lịch sử)"


def separate_rewrite_and_intent(question: str, history):
    """Baseline doi chung: cung hai nhiem vu, tach thanh hai Groq calls."""

    history_text = _history_text(history)

    rewrite_prompt = f"""Bạn là bộ Query Rewriter của chatbot y tế MomCare.

Viết CÂU HỎI MỚI thành một câu độc lập để truy xuất tài liệu.

Quy tắc:
- Giữ nguyên ý định của câu hỏi mới.
- Chỉ lấy từ lịch sử thông tin đang thiếu như đối tượng, độ tuổi hoặc chủ đề.
- Không sao chép câu trả lời cũ vào câu hỏi mới.
- Không thêm mục đích người dùng không hỏi.
- Nếu câu mới đã đầy đủ thì chỉ chuẩn hóa cách diễn đạt.
- Câu viết lại ưu tiên không quá 35 từ.

LỊCH SỬ:
{history_text}

CÂU HỎI MỚI:
{question}

Chỉ trả về câu hỏi đã viết lại.""".strip()

    METER.phase = "rewrite_separate"
    rewritten = _measured_call_llm(
        rewrite_prompt,
        temperature=0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ).strip() or question

    intent_prompt = f"""Bạn là bộ phân loại ý định của chatbot y tế MomCare.

Phân loại CÂU HỎI MỚI vào đúng một trong bốn nhóm:
- RAG: chăm sóc mẹ, thai kỳ, sau sinh, trẻ nhỏ, dinh dưỡng hoặc sức khỏe.
- SMALLTALK: chào hỏi, cảm ơn hoặc trò chuyện xã giao.
- OUT_OF_SCOPE: nội dung ngoài phạm vi chăm sóc mẹ và trẻ.
- BLOCKED: yêu cầu nguy hiểm hoặc không an toàn.

CÂU HỎI MỚI:
{question}

Chỉ trả về một nhãn: RAG/SMALLTALK/OUT_OF_SCOPE/BLOCKED.""".strip()

    METER.phase = "intent_separate"
    raw_intent = _measured_call_llm(
        intent_prompt,
        temperature=0,
        max_tokens=20,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ).strip().upper()

    intent_match = re.search(r"\b(RAG|SMALLTALK|OUT_OF_SCOPE|BLOCKED)\b", raw_intent)
    intent = intent_match.group(1) if intent_match else "UNKNOWN"
    return rewritten, intent


def merged_rewrite_and_intent(question: str, history):
    """Goi dung ham Task Merging hien tai trong llm_chain.py."""
    METER.phase = "rewrite_intent_merged"
    return llm_chain.rewrite_and_detect_intent(question, history)


CONFIGS = {
    "RAW_SEPARATE": {"memory": "raw", "task_merging": False},
    "RAW_MERGED": {"memory": "raw", "task_merging": True},
    "ACM_SEPARATE": {"memory": "acm", "task_merging": False},
    "ACM_MERGED": {"memory": "acm", "task_merging": True},
}


def _calls_for(config: str, turn: int):
    return [r for r in CALL_ROWS if r["config"] == config and r["turn"] == turn]


def run_one_config(config_name: str, turns: int):
    cfg = CONFIGS[config_name]
    history: list[dict[str, str]] = []
    previous_summary = ""
    summarized_count = 0
    rows = []

    for turn_index, (question, assistant_reply) in enumerate(CONTROLLED_TURNS[:turns], start=1):
        METER.config = config_name
        METER.turn = turn_index
        METER.phase = ""
        started = time.perf_counter()
        status = "OK"
        error = ""
        rewritten = ""
        intent = ""
        summary_updated = False

        raw_chars = sum(len(str(m.get("content", ""))) for m in history)
        model_chars = raw_chars

        try:
            if cfg["memory"] == "acm":
                (
                    model_history,
                    previous_summary,
                    summarized_count,
                    summary_updated,
                    raw_chars,
                    model_chars,
                ) = build_acm_history(history, previous_summary, summarized_count)
            else:
                model_history, model_chars = build_raw_history(history)

            if cfg["task_merging"]:
                rewritten, intent = merged_rewrite_and_intent(question, model_history)
            else:
                rewritten, intent = separate_rewrite_and_intent(question, model_history)
        except Exception as exc:  # pragma: no cover - phu thuoc API
            status = "ERROR"
            error = str(exc)

        elapsed = time.perf_counter() - started
        calls = _calls_for(config_name, turn_index)
        summary_tokens = sum(
            int(r["prompt_tokens"]) for r in calls if r["phase"] == "summary_update"
        )
        processing_tokens = sum(
            int(r["prompt_tokens"]) for r in calls if r["phase"] != "summary_update"
        )
        prompt_tokens = sum(int(r["prompt_tokens"]) for r in calls)
        completion_tokens = sum(int(r["completion_tokens"]) for r in calls)

        rows.append(
            {
                "config": config_name,
                "turn": turn_index,
                "question": question,
                "history_chars_raw": raw_chars,
                "history_chars_model": model_chars,
                "summary_updated": summary_updated,
                "api_calls": len(calls),
                "summary_prompt_tokens": summary_tokens,
                "processing_prompt_tokens": processing_tokens,
                "prompt_tokens_total": prompt_tokens,
                "completion_tokens_total": completion_tokens,
                "latency_s": round(elapsed, 6),
                "intent": intent,
                "rewritten": rewritten,
                "status": status,
                "error": error,
            }
        )

        print(
            f"{config_name:<13} | turn={turn_index:02d} | calls={len(calls)} | "
            f"prompt={prompt_tokens:4d} | summary={summary_tokens:4d} | "
            f"history={raw_chars:4d}->{model_chars:4d} chars | {status}"
        )

        # Sau khi xu ly cau hoi hien tai, them cung mot cap Human/AI co dinh
        # vao lich su cua tung cau hinh. Khong goi LLM de sinh response.
        history.append({"type": "human", "content": question})
        history.append({"type": "ai", "content": assistant_reply})

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: list[dict[str, Any]]):
    summaries = []
    for config_name in CONFIGS:
        subset = [r for r in rows if r["config"] == config_name and r["status"] == "OK"]
        if not subset:
            continue

        summaries.append(
            {
                "config": config_name,
                "n_turns_ok": len(subset),
                "total_api_calls": sum(int(r["api_calls"]) for r in subset),
                "mean_api_calls_per_turn": round(_mean([float(r["api_calls"]) for r in subset]), 4),
                "total_prompt_tokens": sum(int(r["prompt_tokens_total"]) for r in subset),
                "mean_prompt_tokens_per_turn": round(
                    _mean([float(r["prompt_tokens_total"]) for r in subset]), 2
                ),
                "mean_summary_prompt_tokens": round(
                    _mean([float(r["summary_prompt_tokens"]) for r in subset]), 2
                ),
                "mean_processing_prompt_tokens": round(
                    _mean([float(r["processing_prompt_tokens"]) for r in subset]), 2
                ),
                "mean_raw_history_chars": round(
                    _mean([float(r["history_chars_raw"]) for r in subset]), 2
                ),
                "mean_model_history_chars": round(
                    _mean([float(r["history_chars_model"]) for r in subset]), 2
                ),
                "mean_latency_s": round(_mean([float(r["latency_s"]) for r in subset]), 4),
            }
        )
    return summaries


def print_summary(summary_rows: list[dict[str, Any]]):
    print("\n" + "=" * 100)
    print("TOKEN EFFICIENCY SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<14} {'Calls':>7} {'Calls/turn':>11} {'Prompt total':>13} "
        f"{'Prompt/turn':>12} {'History chars':>15}"
    )
    print("-" * 100)
    for row in summary_rows:
        history = f"{row['mean_raw_history_chars']:.0f}->{row['mean_model_history_chars']:.0f}"
        print(
            f"{row['config']:<14} {row['total_api_calls']:>7} "
            f"{row['mean_api_calls_per_turn']:>11.2f} "
            f"{row['total_prompt_tokens']:>13} "
            f"{row['mean_prompt_tokens_per_turn']:>12.2f} "
            f"{history:>15}"
        )

    lookup = {row["config"]: row for row in summary_rows}
    if "RAW_SEPARATE" in lookup and "ACM_MERGED" in lookup:
        base = lookup["RAW_SEPARATE"]
        final = lookup["ACM_MERGED"]
        base_tokens = float(base["total_prompt_tokens"])
        final_tokens = float(final["total_prompt_tokens"])
        base_calls = float(base["total_api_calls"])
        final_calls = float(final["total_api_calls"])
        token_change = ((base_tokens - final_tokens) / base_tokens * 100.0) if base_tokens else 0.0
        call_change = ((base_calls - final_calls) / base_calls * 100.0) if base_calls else 0.0
        print("-" * 100)
        print(f"RAW_SEPARATE -> ACM_MERGED: prompt token reduction = {token_change:.2f}%")
        print(f"RAW_SEPARATE -> ACM_MERGED: API call reduction      = {call_change:.2f}%")
        print("Luu y: token reduction am nghia la ACM_MERGED ton nhieu token hon baseline")
        print("trong do dai hoi thoai dang thu. Khong sua/loai bo ket qua nay neu xay ra.")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="So luot hoi thoai co dinh muon benchmark (1-10, mac dinh 10).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help=(
            "Cau hinh can chay. Mac dinh chay ca 4. "
            "Vi du: --configs ACM_MERGED"
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Hau to cho file CSV khi chay retry, vi du: --suffix retry",
    )
    args = parser.parse_args()
    turns = max(1, min(int(args.turns), len(CONTROLLED_TURNS)))

    print("=" * 100)
    print("MOMCARE TOKEN EFFICIENCY BENCHMARK")
    print(f"Turns: {turns} | Model: {getattr(llm_chain, 'MODEL_NAME', '?')}")
    print("Chi do Context Management + Rewrite/Intent; KHONG Retrieval/Generation/Guardrails.")
    print("Lich su la kich ban mo phong co kiem soat va giong nhau cho moi cau hinh.")
    print("=" * 100)

    all_rows = []
    try:
        for config_name in args.configs:
            print(f"\n--- {config_name} ---")
            all_rows.extend(run_one_config(config_name, turns))
    finally:
        llm_chain.call_llm = ORIGINAL_CALL_LLM

    summary_rows = summarize_results(all_rows)
    output_dir = PROJECT_ROOT / "evaluation"
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "", str(args.suffix or "").strip())
    suffix = f"_{suffix}" if suffix else ""
    result_path = output_dir / f"token_efficiency_results{suffix}.csv"
    summary_path = output_dir / f"token_efficiency_summary{suffix}.csv"
    calls_path = output_dir / f"token_efficiency_calls{suffix}.csv"
    write_csv(result_path, all_rows)
    write_csv(summary_path, summary_rows)
    write_csv(calls_path, CALL_ROWS)
    print_summary(summary_rows)

    print(f"\nCSV chi tiet : {result_path}")
    print(f"CSV tong hop : {summary_path}")
    print(f"CSV tung call: {calls_path}")


if __name__ == "__main__":
    main()
