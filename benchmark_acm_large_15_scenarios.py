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

Chạy toàn bộ 15 kịch bản:
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
HISTORY_LIGHT_THRESHOLD = 800
HISTORY_STRONG_THRESHOLD = 1400
HISTORY_MESSAGE_SUMMARY_MIN_CHARS = 150

SLEEP_BETWEEN_METHODS = 2
OUTPUT_PREFIX = "acm_large_15_scenarios"

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
    - dưới 800 ký tự: keep_all;
    - 800–1399 ký tự: light_summary;
    - từ 1400 ký tự: strong_summary;
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



# Hai cặp hội thoại gây nhiễu được lấy từ KB1, dùng thống nhất để đưa
# thông tin cốt lõi ra ngoài cửa sổ 6 tin nhắn của Fixed Window.
DISTRACTOR_PAIRS = [
    (
        "Khi nào thì em cần đưa bé đi khám sức khỏe định kỳ dù bé vẫn đang khỏe mạnh ạ?",
        "Trẻ nên được khám định kỳ để theo dõi cân nặng, chiều cao, các mốc phát triển tâm vận động và nhận tư vấn về dinh dưỡng, tiêm chủng.",
    ),
    (
        "Làm sao để em nhận biết bé đang bắt đầu học hỏi và khám phá thế giới xung quanh?",
        "Trẻ học hỏi qua các giác quan như nhìn, nghe, sờ đồ vật và đưa đồ vật vào miệng để khám phá.",
    ),
]


def make_long_history(
    core_pairs: list[tuple[str, str]],
    bridge_user: str,
    bridge_ai: str,
) -> list[BaseMessage]:
    """
    Tạo lịch sử 10 tin nhắn:
    - 4 tin đầu chứa thông tin cốt lõi;
    - 4 tin giữa là hội thoại gây nhiễu từ KB1;
    - 2 tin cuối chỉ nói chung về việc hỏi tiếp.

    Cách bố trí này giúp kiểm tra trường hợp thông tin cần thiết đã nằm ngoài
    cửa sổ 6 tin nhắn gần nhất nhưng vẫn còn trong giới hạn 20 tin của ACM.
    """
    history: list[BaseMessage] = []

    for user_text, ai_text in core_pairs:
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=ai_text))

    for user_text, ai_text in DISTRACTOR_PAIRS:
        history.append(HumanMessage(content=user_text))
        history.append(AIMessage(content=ai_text))

    history.append(HumanMessage(content=bridge_user))
    history.append(AIMessage(content=bridge_ai))

    return history


SCENARIOS: dict[str, Scenario] = {
    "vitamin_d": Scenario(
        scenario_id="vitamin_d",
        title="Nguồn bổ sung vitamin D cho trẻ 6 tháng bú mẹ",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi hiện được 6 tháng tuổi và vẫn bú mẹ hoàn toàn. Vì sao bé cần bổ sung vitamin D?",
                    "Hàm lượng vitamin D trong sữa mẹ thấp và không đáp ứng đủ nhu cầu của trẻ. Nhu cầu khuyến nghị đối với trẻ nhỏ dưới 6 tháng là 400 UI mỗi ngày.",
                ),
                (
                    "Thiếu vitamin D có thể ảnh hưởng như thế nào đến bé?",
                    "Thiếu vitamin D có thể gây còi xương, chiều cao thấp, biến dạng xương hoặc chậm mọc răng.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về nguồn cung cấp của loại vitamin đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dựa vào phần lịch sử đã được quản lý để xác định loại vitamin và độ tuổi của trẻ.",
        ),
        question="Loại vitamin đó còn có thể được bổ sung từ những nguồn nào?",
        required_groups=[
            ["vitamin d"],
            ["6 tháng", "trẻ 6 tháng"],
            ["nguồn bổ sung", "bổ sung từ", "nguồn nào", "nguồn cung cấp"],
        ],
        source_note="KB1 câu 398--400.",
    ),

    "oral_care": Scenario(
        scenario_id="oral_care",
        title="Dụng cụ vệ sinh miệng cho trẻ 6 tháng chưa mọc răng",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi được 6 tháng tuổi, chưa mọc răng rõ và vẫn bú bình thường. Em nên vệ sinh răng miệng cho bé từ khi nào?",
                    "Nên vệ sinh ngay từ khi bé chưa mọc răng bằng gạc mềm thấm nước sạch.",
                ),
                (
                    "Khi răng bắt đầu mọc thì cần đổi dụng cụ như thế nào?",
                    "Khi răng bắt đầu mọc, hãy dùng bàn chải lông mềm dành riêng cho trẻ nhỏ.",
                ),
            ],
            bridge_user="Tôi muốn hỏi lại phần dụng cụ chăm sóc vùng miệng đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ sử dụng ngữ cảnh trước đó để xác định đúng độ tuổi và dụng cụ đang được hỏi.",
        ),
        question="Còn khi bé chưa mọc răng thì nên dùng dụng cụ gì để vệ sinh?",
        required_groups=[
            ["vệ sinh răng miệng", "vệ sinh miệng", "làm sạch lợi", "chăm sóc vùng miệng"],
            ["6 tháng", "trẻ 6 tháng"],
            ["gạc mềm", "khăn mềm", "dụng cụ"],
        ],
        source_note="KB1 câu 196 và 198.",
    ),

    "complementary_feeding": Scenario(
        scenario_id="complementary_feeding",
        title="Độ đặc và lượng ăn dặm cho trẻ 6 tháng",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi vừa tròn 6 tháng tuổi và đang bắt đầu ăn dặm. Khi nào trẻ nên bắt đầu giai đoạn ăn bổ sung?",
                    "Trẻ bắt đầu ăn bổ sung khi tròn 6 tháng tuổi, tương đương 180 ngày.",
                ),
                (
                    "Nguyên tắc ban đầu về độ đặc và số lượng thức ăn là gì?",
                    "Cho trẻ ăn từ loãng đến đặc, từ ít đến nhiều và tiếp tục bú mẹ.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp nguyên tắc về kết cấu và lượng thức ăn đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để khôi phục đúng độ tuổi và chủ đề ăn bổ sung.",
        ),
        question="Còn độ đặc và số lượng thì áp dụng nguyên tắc nào cho bé?",
        required_groups=[
            ["ăn bổ sung", "ăn dặm"],
            ["6 tháng", "trẻ 6 tháng"],
            ["độ đặc", "loãng", "số lượng", "lượng thức ăn"],
        ],
        source_note="KB1 câu 19--20 và 289--292.",
    ),

    "vomiting_danger": Scenario(
        scenario_id="vomiting_danger",
        title="Dấu hiệu cần đi khám khi trẻ 8 tháng nôn ói",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 8 tháng tuổi bị nôn ói nhiều. Những nguyên nhân thường gặp ở trẻ dưới 12 tháng là gì?",
                    "Nguyên nhân có thể gồm trào ngược dạ dày thực quản, bệnh lý ngoại khoa, nhiễm trùng hoặc tư thế bú và ăn dặm chưa đúng.",
                ),
                (
                    "Những dấu hiệu nào cho thấy tình trạng nôn ói cần đưa bé đi khám ngay?",
                    "Cần đi khám khi trẻ nôn kéo dài, dịch nôn có máu hoặc màu vàng xanh, mất nước, sốt cao, li bì, co giật hoặc bỏ bú.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về các dấu hiệu nguy hiểm của tình trạng vừa trao đổi.",
            bridge_ai="MomCare sẽ dùng lịch sử để xác định đúng triệu chứng và độ tuổi của trẻ.",
        ),
        question="Còn khi nào thì tình trạng đó cần đưa bé đi khám ngay?",
        required_groups=[
            ["nôn ói", "nôn", "ói"],
            ["8 tháng", "trẻ 8 tháng"],
            ["đi khám", "khám ngay", "dấu hiệu nguy hiểm"],
        ],
        source_note="KB1 câu 188--189 và 395--397.",
    ),

    "diarrhea_rehydration": Scenario(
        scenario_id="diarrhea_rehydration",
        title="Bù nước khi trẻ 2 tuổi bị tiêu chảy",
        history=make_long_history(
            core_pairs=[
                (
                    "Con tôi 2 tuổi đang bị tiêu chảy. Có nên cho bé nhịn ăn để bụng mau lành không?",
                    "Không được cho trẻ nhịn ăn; cần tiếp tục cho bú hoặc ăn thức ăn dễ tiêu.",
                ),
                (
                    "Tại nhà cần cho bé uống gì để phòng mất nước?",
                    "Cho trẻ uống Oresol theo hướng dẫn; có thể tiếp tục bú mẹ và dùng nước cháo hoặc nước cơm phù hợp.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về loại dung dịch bù nước đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để xác định bệnh, độ tuổi và dung dịch đang được nhắc tới.",
        ),
        question="Còn loại dung dịch đó nên dùng để xử trí tình trạng nào của bé?",
        required_groups=[
            ["tiêu chảy"],
            ["2 tuổi", "trẻ 2 tuổi"],
            ["oresol", "ô-rê-zôn", "bù nước"],
        ],
        source_note="KB1 câu 25 và 187--189.",
    ),

    "breastmilk_storage": Scenario(
        scenario_id="breastmilk_storage",
        title="Bảo quản sữa mẹ trong ngăn mát",
        history=make_long_history(
            core_pairs=[
                (
                    "Tôi đang vắt sữa cho bé 3 tháng tuổi. Sữa mẹ để ở nhiệt độ thường được bao lâu?",
                    "Sữa mẹ vắt ra chỉ nên để ở nhiệt độ thường khoảng 3 giờ.",
                ),
                (
                    "Nếu chuyển sữa vào ngăn mát dưới 4 độ C thì bảo quản được bao lâu?",
                    "Sữa mẹ trong ngăn mát dưới 4 độ C tốt nhất dùng trong 4 ngày và có thể để tới 8 ngày.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về thời gian bảo quản trong môi trường lạnh vừa nói.",
            bridge_ai="MomCare sẽ dùng lịch sử để xác định loại sữa, nhiệt độ bảo quản và độ tuổi của bé.",
        ),
        question="Còn trong ngăn mát thì loại sữa đó được giữ trong bao lâu?",
        required_groups=[
            ["sữa mẹ"],
            ["3 tháng", "trẻ 3 tháng", "bé 3 tháng"],
            ["ngăn mát", "4 ngày", "8 ngày"],
        ],
        source_note="KB1 câu 16 và 42--44.",
    ),

    "newborn_danger_signs": Scenario(
        scenario_id="newborn_danger_signs",
        title="Dấu hiệu nguy hiểm ở trẻ sơ sinh 10 ngày tuổi",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi mới 10 ngày tuổi. Gia đình cần theo dõi những dấu hiệu nguy hiểm nào?",
                    "Cần theo dõi bú kém hoặc bỏ bú, ngủ li bì khó đánh thức, thở bất thường, sốt hoặc hạ nhiệt độ, co giật và các biểu hiện bất thường khác.",
                ),
                (
                    "Nếu trẻ sốt cao tại nhà thì xử trí ban đầu thế nào?",
                    "Lau người bằng khăn ấm, giữ trẻ thoáng và đưa trẻ đến cơ sở y tế.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về nhóm dấu hiệu cần đưa trẻ đi khám ngay.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để xác định trẻ sơ sinh và nhóm dấu hiệu đang được nhắc tới.",
        ),
        question="Còn những biểu hiện nào của bé được xem là dấu hiệu nguy hiểm?",
        required_groups=[
            ["dấu hiệu nguy hiểm", "nguy hiểm"],
            ["10 ngày", "trẻ sơ sinh"],
            ["bú kém", "bỏ bú", "li bì", "khó thở", "co giật"],
        ],
        source_note="KB1 câu 6, 9 và 10.",
    ),

    "bcg_lymphadenitis": Scenario(
        scenario_id="bcg_lymphadenitis",
        title="Viêm hạch sau tiêm BCG ở trẻ 2 tháng",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 2 tháng tuổi đã tiêm ngừa lao BCG. Viêm hạch bạch huyết sau tiêm là gì?",
                    "Đây là một phản ứng bất lợi có thể gặp sau tiêm BCG, thường xuất hiện ở vùng hạch gần chỗ tiêm.",
                ),
                (
                    "Biểu hiện thường gặp của tình trạng này là gì?",
                    "Hạch có thể sưng tròn trên 1 cm, cùng bên chỗ tiêm, thường ở nách, sau tai hoặc cổ; một số trường hợp có thể sưng đỏ hoặc chảy mủ.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về cách xử trí phản ứng sau mũi tiêm đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dùng lịch sử để nhận diện loại vắc xin, phản ứng và độ tuổi của trẻ.",
        ),
        question="Còn phản ứng nổi hạch đó sau mũi tiêm nào và nên xử trí ra sao?",
        required_groups=[
            ["bcg", "tiêm ngừa lao", "vắc xin lao"],
            ["2 tháng", "trẻ 2 tháng"],
            ["viêm hạch", "nổi hạch", "hạch"],
        ],
        source_note="KB1 câu 392--394.",
    ),

    "exclusive_breastfeeding_water": Scenario(
        scenario_id="exclusive_breastfeeding_water",
        title="Có cần cho trẻ 4 tháng bú mẹ hoàn toàn uống nước",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 4 tháng tuổi đang bú mẹ hoàn toàn. Nuôi con bằng sữa mẹ hoàn toàn nghĩa là gì?",
                    "Trẻ chỉ bú sữa mẹ mà không ăn hoặc uống thêm thứ khác, trừ trường hợp có chỉ định phù hợp.",
                ),
                (
                    "Sữa mẹ chứa khoảng bao nhiêu phần trăm nước và bé có cần uống thêm nước trắng không?",
                    "Sữa mẹ chứa khoảng 88 phần trăm nước nên trẻ bú mẹ hoàn toàn không cần uống thêm nước trắng.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về loại nước có cần bổ sung cho bé trong giai đoạn này.",
            bridge_ai="MomCare sẽ dùng lịch sử để xác định chế độ nuôi dưỡng và độ tuổi của trẻ.",
        ),
        question="Còn loại nước đó có cần cho bé uống thêm không?",
        required_groups=[
            ["sữa mẹ", "bú mẹ hoàn toàn"],
            ["4 tháng", "trẻ 4 tháng"],
            ["nước trắng", "uống thêm nước", "không cần"],
        ],
        source_note="KB1 câu 7, 34 và 69--70.",
    ),

    "stomach_capacity": Scenario(
        scenario_id="stomach_capacity",
        title="Dung tích dạ dày trẻ sơ sinh ngày thứ ba",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi đang ở ngày đầu sau sinh. Dung tích dạ dày mỗi lần bú khoảng bao nhiêu?",
                    "Trong ngày đầu, dung tích dạ dày của trẻ khoảng 5 đến 7 ml mỗi lần bú.",
                ),
                (
                    "Khi sang ngày thứ ba thì dung tích dạ dày tăng lên mức nào?",
                    "Đến ngày thứ ba sau sinh, dung tích dạ dày tăng lên khoảng 22 đến 27 ml mỗi lần bú.",
                ),
            ],
            bridge_user="Tôi muốn hỏi lại con số ở mốc thời gian thứ hai đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để xác định mốc ngày sau sinh và đại lượng đang được hỏi.",
        ),
        question="Còn ở mốc đó thì mỗi lần bú dạ dày của bé chứa khoảng bao nhiêu?",
        required_groups=[
            ["dung tích dạ dày", "dạ dày"],
            ["ngày thứ 3", "ngày thứ ba", "3 ngày"],
            ["22", "27", "ml"],
        ],
        source_note="KB1 câu 31--33.",
    ),

    "effective_breastfeeding": Scenario(
        scenario_id="effective_breastfeeding",
        title="Dấu hiệu ngậm bắt vú đúng ở trẻ 1 tháng",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 1 tháng tuổi bú mẹ. Tư thế bú đúng cần bảo đảm những điểm nào?",
                    "Mẹ ở tư thế thoải mái, trẻ được giữ sát người mẹ và đầu, thân trẻ nằm trên một đường thẳng.",
                ),
                (
                    "Dấu hiệu nào cho thấy bé ngậm bắt vú và bú hiệu quả?",
                    "Miệng trẻ mở rộng, ngậm sâu quầng vú, môi dưới trễ ra ngoài và trẻ bú không gây đau cho mẹ.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về biểu hiện cho thấy kỹ thuật vừa nói đang đúng.",
            bridge_ai="MomCare sẽ dùng lịch sử để xác định kỹ thuật bú và độ tuổi của trẻ.",
        ),
        question="Còn dấu hiệu nào cho thấy bé thực hiện đúng cách đó?",
        required_groups=[
            ["bú mẹ", "ngậm bắt vú", "bú hiệu quả"],
            ["1 tháng", "trẻ 1 tháng"],
            ["miệng", "quầng vú", "môi dưới"],
        ],
        source_note="KB1 câu 14--15, 39 và 71.",
    ),

    "child_constipation": Scenario(
        scenario_id="child_constipation",
        title="Điều chỉnh ăn uống khi trẻ 3 tuổi táo bón",
        history=make_long_history(
            core_pairs=[
                (
                    "Con tôi 3 tuổi thường táo bón và đi ngoài khó khăn. Nên điều chỉnh chế độ ăn thế nào?",
                    "Nên tăng thực phẩm giàu chất xơ như rau xanh, đu đủ hoặc mận, cho trẻ uống đủ nước và khuyến khích vận động.",
                ),
                (
                    "Việc uống đủ nước và vận động có cần duy trì thường xuyên không?",
                    "Có, đây là những biện pháp hỗ trợ nhu động ruột và hạn chế phân khô cứng.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về nhóm thực phẩm đã được khuyên dùng cho tình trạng ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để xác định tình trạng, độ tuổi và nhóm thực phẩm.",
        ),
        question="Còn nhóm thực phẩm đó nên tăng cho bé để cải thiện tình trạng gì?",
        required_groups=[
            ["táo bón"],
            ["3 tuổi", "trẻ 3 tuổi"],
            ["chất xơ", "rau xanh", "trái cây"],
        ],
        source_note="KB1 câu 190.",
    ),

    "newborn_jaundice": Scenario(
        scenario_id="newborn_jaundice",
        title="Phân biệt vàng da sinh lý và bệnh lý ở trẻ 5 ngày tuổi",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 5 ngày tuổi đang bị vàng da. Vàng da sinh lý thường xuất hiện khi nào?",
                    "Vàng da sinh lý thường xuất hiện sau 24 giờ tuổi, mức độ nhẹ và không kèm triệu chứng bất thường.",
                ),
                (
                    "Dấu hiệu nào gợi ý vàng da bệnh lý cần đi khám?",
                    "Vàng da xuất hiện sớm trong 1 đến 2 ngày đầu, lan rộng, kéo dài hoặc kèm bỏ bú, nôn, sốt, khóc nhiều hay phân bạc màu cần được khám.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về loại vàng da nguy hiểm hơn đã nói ở đầu cuộc trò chuyện.",
            bridge_ai="MomCare sẽ sử dụng lịch sử để xác định loại vàng da và tuổi của trẻ.",
        ),
        question="Còn loại đó có những dấu hiệu nào cho thấy bé cần đi khám?",
        required_groups=[
            ["vàng da bệnh lý", "vàng da"],
            ["5 ngày", "trẻ 5 ngày"],
            ["đi khám", "xuất hiện sớm", "bỏ bú", "lan"],
        ],
        source_note="KB1 câu 153.",
    ),

    "child_fever": Scenario(
        scenario_id="child_fever",
        title="Ngưỡng sốt và dấu hiệu nguy hiểm ở trẻ 18 tháng",
        history=make_long_history(
            core_pairs=[
                (
                    "Bé nhà tôi 18 tháng tuổi. Nhiệt độ nách từ mức nào được xem là sốt?",
                    "Trẻ được coi là sốt khi nhiệt độ nách từ 37,5 độ C trở lên.",
                ),
                (
                    "Khi nào sốt được xem là nguy hiểm và cần đưa bé đi khám?",
                    "Cần đi khám khi sốt kèm li bì, khó thở, co giật hoặc sốt cao không hạ.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về ngưỡng nhiệt độ và các dấu hiệu cảnh báo đã nói ở đầu.",
            bridge_ai="MomCare sẽ dựa vào lịch sử để xác định độ tuổi và tình trạng sốt.",
        ),
        question="Còn mức nhiệt đó kèm biểu hiện nào thì phải đưa bé đi khám?",
        required_groups=[
            ["sốt"],
            ["18 tháng", "trẻ 18 tháng"],
            ["37,5", "li bì", "khó thở", "co giật", "đi khám"],
        ],
        source_note="KB1 câu 10 và 185.",
    ),

    "postpartum_urinary_retention": Scenario(
        scenario_id="postpartum_urinary_retention",
        title="Xử trí bí tiểu sau sinh",
        history=make_long_history(
            core_pairs=[
                (
                    "Tôi sinh con được 6 giờ nhưng vẫn chưa đi tiểu được. Đây có thể là tình trạng gì?",
                    "Đây có thể là tiểu tồn lưu hoặc bí tiểu sau sinh và cần báo nhân viên y tế.",
                ),
                (
                    "Có biện pháp nào hỗ trợ tại nhà để dễ đi tiểu hơn không?",
                    "Nên vận động sớm và tập đi tiểu mỗi 3 giờ một lần dù có hoặc chưa có cảm giác buồn tiểu.",
                ),
            ],
            bridge_user="Tôi muốn hỏi tiếp về biện pháp hỗ trợ cho tình trạng sau sinh đã nói ở đầu.",
            bridge_ai="MomCare sẽ dùng lịch sử để xác định đối tượng là mẹ sau sinh và vấn đề đang được hỏi.",
        ),
        question="Còn biện pháp đó được dùng để hỗ trợ tình trạng nào sau sinh?",
        required_groups=[
            ["bí tiểu", "tiểu tồn lưu"],
            ["sau sinh", "sau khi sinh"],
            ["vận động sớm", "3 giờ", "tập tiểu"],
        ],
        source_note="KB1 câu 104--108.",
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

        # Nếu có nhiều hơn một lần gọi:
        # các lần đầu là tóm tắt, lần cuối là Query Rewriting.
        "summary_tokens": (
            sum(tokens[:-1])
            if len(tokens) > 1
            else 0
        ),

        "rewrite_tokens": (
            tokens[-1]
            if tokens
            else 0
        ),

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
            f"{row['history_reduction_percent']:.1f}% | "
            f"mode={row['memory_mode']}"
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
            f"summary tokens: {row['summary_tokens']} | "
            f"rewrite tokens: {row['rewrite_tokens']} | "
            f"total tokens: {row['api_tokens_logged_total']} | "
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
            avg_api_calls=(
                "api_calls_logged",
                "mean",
            ),
            avg_summary_tokens=(
                "summary_tokens",
                "mean",
            ),
            avg_rewrite_tokens=(
                "rewrite_tokens",
                "mean",
            ),
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
        "avg_summary_tokens",
        "avg_rewrite_tokens",
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
        help="Chạy toàn bộ 15 kịch bản.",
    )
    args = parser.parse_args()

    selected = (
        list(SCENARIOS.values())
        if args.all
        else [SCENARIOS[args.scenario]]
    )

    detail_frames: list[pd.DataFrame] = []

    for scenario_index, scenario in enumerate(selected, start=1):
        print(
            f"\n>>> Đang chạy kịch bản "
            f"{scenario_index}/{len(selected)}: "
            f"{scenario.scenario_id}"
        )

        detail_frames.append(run_scenario(scenario))

        # Lưu tạm sau từng kịch bản để không mất toàn bộ kết quả
        # nếu mạng hoặc API bị gián đoạn giữa quá trình chạy dài.
        partial_detail = pd.concat(
            detail_frames,
            ignore_index=True,
        )
        partial_detail.to_csv(
            f"{OUTPUT_PREFIX}_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    detail = pd.concat(
        detail_frames,
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
