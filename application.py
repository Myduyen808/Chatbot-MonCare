from unittest import result

import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
import yaml
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

from history_handle import CustomHistory, get_list_names, get_history_id, get_chat_messages
import llm_chain
from vectordb import get_list_documents, get_document, delete_document, get_details, create_vectordb_with_file

DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.0

FEEDBACK_LOG_PATH = Path("runtime_logs") / "user_feedback.csv"

# =========================================================
# CẤU HÌNH ROLLING ADAPTIVE MEMORY
# =========================================================

# Hai tin nhắn gần nhất luôn được giữ nguyên:
# một câu hỏi người dùng và một phản hồi của MomCare.
HISTORY_KEEP_RECENT = 2

# Khi có ít nhất hai tin nhắn cũ chưa được đưa vào summary,
# hệ thống cập nhật bản tóm tắt tích lũy.
#
# Hai tin nhắn tương ứng với một lượt Human + AI hoàn chỉnh.
ROLLING_SUMMARY_MIN_MESSAGES = 2

# Không tạo summary nếu phần tin nhắn cũ quá ngắn.
# Ngưỡng này tránh gọi LLM cho những lượt trao đổi rất ngắn.
ROLLING_SUMMARY_TRIGGER_CHARS = 250

# =========================================================
# PHẢN HỒI KHÔNG ĐƯỢC DÙNG LÀM NGỮ CẢNH HỘI THOẠI
# =========================================================
BLOCKED_RESPONSE_MARKERS = [
    "MomCare không thể hỗ trợ yêu cầu này",
    "MomCare không thể xử lý yêu cầu này",
    "MomCare không thể tư vấn về các sản phẩm không rõ nguồn gốc",
    "DỪNG LẠI! Hành động này rất nguy hiểm",
    "MomCare không thể cung cấp thông tin kê đơn",
    "MomCare không thể tư vấn về liều lượng thuốc",
    "CẢNH BÁO: Tình trạng của mẹ cần được xử lý Y TẾ NGAY",
    "⚠️ Hệ thống đang gặp sự cố kỹ thuật",
    "⚠️ Hệ thống AI đang gặp sự cố kết nối",

    # Phản hồi hỗ trợ sức khỏe tinh thần
    "Mẹ không đơn độc đâu",
    "Đường dây hỗ trợ sức khỏe tinh thần",
    "hãy để người khác giúp mẹ",
]


def is_blocked_or_safety_response(text: str) -> bool:
    """Kiểm tra phản hồi từ chối hoặc cảnh báo an toàn."""
    if not text:
        return False

    normalized_text = text.lower()

    return any(
        marker.lower() in normalized_text
        for marker in BLOCKED_RESPONSE_MARKERS
    )

def build_adaptive_history(
    filtered_history,
    previous_summary: str,
    summarized_count: int,
):
    """
    Xây dựng ngữ cảnh theo cơ chế Rolling Summary.

    Trạng thái đầu vào:
    - previous_summary: bản tóm tắt tích lũy hiện có;
    - summarized_count: số tin nhắn an toàn đã được đưa vào summary.

    Quy tắc:
    - Luôn giữ nguyên hai tin nhắn gần nhất.
    - Chỉ xử lý phần tin nhắn cũ chưa từng được tóm tắt.
    - Nếu phần mới đủ điều kiện, cập nhật summary cũ.
    - Không tóm tắt lại toàn bộ lịch sử từ đầu.
    """
    safe_history = list(filtered_history or [])
    input_messages = len(safe_history)

    total_chars_before = sum(
        len(str(message.get("content", "")))
        for message in safe_history
    )

    if not safe_history:
        return (
            [],
            {
                "mode": "empty",
                "total_chars_before": 0,
                "total_chars_after": 0,
                "input_messages": 0,
                "output_messages": 0,
                "summarized_count_before": 0,
                "summarized_count_after": 0,
            },
            "",
            0,
        )

    # Tránh sai chỉ số khi đổi hoặc tải lại phiên hội thoại.
    if (
        summarized_count < 0
        or summarized_count > input_messages
    ):
        summarized_count = 0
        previous_summary = ""

    # Hai tin nhắn gần nhất chưa đưa vào summary.
    split_index = max(
        0,
        input_messages - HISTORY_KEEP_RECENT
    )

    # Nếu số tin đã tóm tắt lớn hơn vùng lịch sử cũ hiện có,
    # reset trạng thái để tránh bỏ sót dữ liệu.
    if summarized_count > split_index:
        summarized_count = 0
        previous_summary = ""

    # Chỉ lấy phần lịch sử cũ chưa từng được tóm tắt.
    pending_messages = safe_history[
        summarized_count:split_index
    ]

    pending_chars = sum(
        len(str(message.get("content", "")))
        for message in pending_messages
    )

    enough_messages = (
        len(pending_messages)
        >= ROLLING_SUMMARY_MIN_MESSAGES
    )

    enough_chars = (
        pending_chars
        >= ROLLING_SUMMARY_TRIGGER_CHARS
    )

    updated_summary = previous_summary
    updated_count = summarized_count
    summary_updated = False

    # Chỉ cập nhật khi có ít nhất một lượt hoàn chỉnh và phần cũ
    # không quá ngắn.
    if enough_messages and enough_chars:
        candidate_summary = (
            llm_chain.update_rolling_summary(
                previous_summary=previous_summary,
                new_messages=pending_messages,
            )
        )

        if candidate_summary:
            updated_summary = candidate_summary
            updated_count = split_index
            summary_updated = True

    processed_messages = []

    # Nếu đã có summary thì đưa summary vào đầu ngữ cảnh.
    if updated_summary:
        processed_messages.append(
            {
                "type": "ai",
                "content": (
                    "[Tóm tắt lịch sử cũ] "
                    + updated_summary
                ),
            }
        )

    # Những tin nhắn chưa được đưa vào summary vẫn phải được giữ.
    #
    # Nếu vừa cập nhật summary:
    #   bắt đầu từ updated_count, thường chỉ còn hai tin gần nhất.
    #
    # Nếu chưa đủ điều kiện cập nhật:
    #   giữ phần pending và hai tin gần nhất để không mất ngữ cảnh.
    remaining_messages = safe_history[updated_count:]

    for message in remaining_messages:
        processed_messages.append(
            {
                "type": message.get("type", ""),
                "content": str(
                    message.get("content", "")
                ),
            }
        )

    # Nếu chưa có summary và chưa có gì được tóm tắt,
    # toàn bộ lịch sử ngắn được giữ nguyên.
    if not updated_summary and updated_count == 0:
        mode = "keep_all"
    elif summary_updated and not previous_summary:
        mode = "rolling_summary_create"
    elif summary_updated:
        mode = "rolling_summary_update"
    else:
        mode = "rolling_summary_reuse"

    chat_history_messages = []

    for message in processed_messages:
        content = str(message.get("content", ""))

        if message.get("type") == "human":
            chat_history_messages.append(
                HumanMessage(content=content)
            )
        else:
            chat_history_messages.append(
                AIMessage(content=content)
            )

    total_chars_after = sum(
        len(message.content)
        for message in chat_history_messages
    )

    history_stats = {
        "mode": mode,
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "input_messages": input_messages,
        "output_messages": len(chat_history_messages),
        "pending_messages": len(pending_messages),
        "pending_chars": pending_chars,
        "summarized_count_before": summarized_count,
        "summarized_count_after": updated_count,
    }

    return (
        chat_history_messages,
        history_stats,
        updated_summary,
        updated_count,
    )

with open("db_config.yml", "r", encoding="utf-8") as f:
    db_config = yaml.safe_load(f)

st.set_page_config(page_title="MomCare - Trợ lý Mẹ bỉm sữa", page_icon="🤱", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} 
.main { background-color: #fdf6f0; }
[data-testid="stChatMessage"] {background-color: #ffffff; border-radius: 15px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 5px solid #ff6b81;}
[data-testid="stTextInput"] {border-radius: 20px; border: 2px solid #ff6b81; padding: 10px;}
section[data-testid="stSidebar"] {background-color: #ffffff; box-shadow: 2px 0 10px rgba(0,0,0,0.05);}
.stButton > button {background-color: #ff6b81; color: white; border-radius: 20px; border: none; font-weight: bold;}
.stButton > button:hover {background-color: #ff4757; transform: translateY(-2px);}
</style>
""", unsafe_allow_html=True)

# ── Menu chính ──────────────────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        "Menu Chính",
        ["Chatbot", "Quản lý Dữ liệu"],
        icons=["chat", "database"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"font-family": "sans-serif"},
            "nav-link-selected": {"background-color": "#ff4b4b"},
        },
    )

def clear_cache():
    st.cache_resource.clear()

def rag_click():
    st.session_state.rag_chat = True

@st.cache_resource
def load_chain(rag_chat, number_of_documents):
    if rag_chat:
        return llm_chain.load_rag_chain(number_of_documents)
    return llm_chain.load_normal_chain()

def load_history(history_name):
    history_id = get_history_id(history_name)
    if history_name != "New Session" and history_id is not None:
        history = CustomHistory()
        history.load(history_id=history_id)
        return history
    return CustomHistory()

def render_source_documents(source_docs):
    """Hiển thị danh sách tài liệu tham khảo của câu trả lời RAG."""
    with st.expander(
        "📎 Xem nguồn tài liệu",
        expanded=False,
    ):
        if not source_docs:
            st.warning(
                "Không có tài liệu tham khảo được trả về "
                "cho câu trả lời này."
            )
            return

        st.success(
            f"✅ Tìm thấy {len(source_docs)} nguồn tài liệu"
        )

        for i, doc in enumerate(source_docs, start=1):
            # Hỗ trợ cả LangChain Document và dictionary
            # đã được lưu trong session_state.
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {}) or {}
                page_content = str(
                    doc.get("page_content", "")
                ).strip()
            else:
                metadata = getattr(doc, "metadata", None) or {}
                page_content = str(
                    getattr(doc, "page_content", "")
                ).strip()

            source_name = metadata.get(
                "source",
                metadata.get("file_name", "Không rõ"),
            )

            page = metadata.get(
                "page_display",
                metadata.get("page", "Không xác định"),
            )

            chunk_id = metadata.get(
                "chunk_id",
                "Không xác định",
            )

            file_type = metadata.get(
                "file_type",
                "Không xác định",
            )

            if page_content:
                preview = page_content[:500]

                if len(page_content) > 500:
                    preview += "..."
            else:
                preview = "Không có nội dung xem trước."

            st.info(
                f"**📄 Nguồn {i}:** `{source_name}`  \n"
                f"**Loại:** `{file_type}` | "
                f"**Trang:** `{page}` | "
                f"**Chunk:** `{chunk_id}`  \n\n"
                f"{preview}"
            )


def _ensure_feedback_log():
    """Tạo tệp lưu phản hồi người dùng nếu chưa tồn tại."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_LOG_PATH.exists():
        with FEEDBACK_LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "feedback_id",
                    "rating",
                    "question",
                    "answer_preview",
                    "source_count",
                    "response_time_seconds",
                ]
            )


def _save_user_feedback(
    feedback_id: str,
    rating: str,
    question: str,
    answer: str,
    source_count: int,
    response_time_seconds: float,
):
    """Ghi phản hồi Hữu ích/Chưa hữu ích vào CSV."""
    _ensure_feedback_log()

    with FEEDBACK_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                feedback_id,
                rating,
                question,
                str(answer or "")[:500],
                int(source_count),
                round(float(response_time_seconds), 3),
            ]
        )


def _make_feedback_id(question: str, answer: str) -> str:
    """Tạo định danh ổn định cho một lượt trả lời."""
    payload = f"{question}|{answer}".encode(
        "utf-8",
        errors="ignore",
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def render_feedback_controls():
    """Hiển thị nút đánh giá cho phản hồi gần nhất."""
    feedback_id = st.session_state.get(
        "last_feedback_id",
        "",
    )

    if not feedback_id:
        return

    selected_rating = st.session_state.feedback_by_id.get(
        feedback_id
    )

    st.caption("Phản hồi này có hữu ích không?")

    col_helpful, col_not_helpful, col_status = st.columns(
        [1, 1, 2]
    )

    with col_helpful:
        helpful_clicked = st.button(
            "👍 Hữu ích",
            key=f"helpful_{feedback_id}",
            use_container_width=True,
            disabled=(selected_rating is not None),
        )

    with col_not_helpful:
        not_helpful_clicked = st.button(
            "👎 Chưa hữu ích",
            key=f"not_helpful_{feedback_id}",
            use_container_width=True,
            disabled=(selected_rating is not None),
        )

    rating = None

    if helpful_clicked:
        rating = "helpful"
    elif not_helpful_clicked:
        rating = "not_helpful"

    if rating is not None:
        st.session_state.feedback_by_id[
            feedback_id
        ] = rating

        _save_user_feedback(
            feedback_id=feedback_id,
            rating=rating,
            question=st.session_state.get(
                "last_feedback_question",
                "",
            ),
            answer=st.session_state.get(
                "last_feedback_answer",
                "",
            ),
            source_count=st.session_state.get(
                "last_source_count",
                0,
            ),
            response_time_seconds=st.session_state.get(
                "last_response_time_seconds",
                0.0,
            ),
        )

        selected_rating = rating

    with col_status:
        if selected_rating == "helpful":
            st.success("Đã ghi nhận: Hữu ích")
        elif selected_rating == "not_helpful":
            st.info("Đã ghi nhận: Chưa hữu ích")


# ════════════════════════════════════════════════════════════════════════════
def Chatbot():
    # ── Khởi tạo toàn bộ session_state tại đây ──────────────────────────────
    defaults = {
        "clear_input": False,
        "send_input": False,
        "number_of_documents": DEFAULT_TOP_K,
        "rag_chat": True,
        "history_choice": "New Session",
        "locked_session": False,
        "user_question": "",
        "previous_turn_blocked": False,

        "acm_rolling_summary": "",
        "acm_summarized_count": 0,
        "acm_memory_mode": "keep_all",

        # Lưu nguồn qua lần st.rerun().
        "last_source_docs": [],
        "last_source_question": "",

        # Thông tin vận hành của phản hồi gần nhất.
        "last_response_time_seconds": 0.0,
        "last_source_count": 0,

        # Trạng thái đánh giá câu trả lời.
        "feedback_by_id": {},
        "last_feedback_id": "",
        "last_feedback_question": "",
        "last_feedback_answer": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-size: 50px;">🤱</span>
        <h1 style="color: #ff4757; margin-top: -10px;">MomCare</h1>
        <p style="color: #636e72; font-size: 16px; margin-top: -10px;">Trợ lý AI Chuyên gia Chăm sổ tay Mẹ và Bé</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt Hệ thống")
        # Temperature cố định để đảm bảo tính nhất quán
        # và khả năng tái lập trong chatbot y tế.
        temperature = DEFAULT_TEMPERATURE
        rag_button = st.toggle("Bật Chế độ RAG", value=st.session_state.rag_chat)

        if rag_button != st.session_state.rag_chat:
            st.session_state.rag_chat = rag_button
            st.rerun()

        if st.session_state.rag_chat:
            st.info("🟢 Đang dùng tài liệu.")
            st.session_state.number_of_documents = DEFAULT_TOP_K
            st.caption(
                "Hệ thống sử dụng 5 đoạn tài liệu phù hợp nhất sau bước tái xếp hạng."
            )
        else:
            st.warning("🟠 Đang dùng kiến thức chung.")

        st.markdown("---")

        list_chat_sessions = ["New Session"] + get_list_names()

        if st.session_state.locked_session:
            st.selectbox(
                "🔒 Phiên hiện tại",
                list_chat_sessions,
                index=list_chat_sessions.index(st.session_state.history_choice)
                      if st.session_state.history_choice in list_chat_sessions else 0,
                disabled=True,
                key="chat_session_locked"
            )
            if st.button("➕ Cuộc trò chuyện mới"):
                st.session_state.locked_session = False
                st.session_state.history_choice = "New Session"
                st.session_state.current_history = CustomHistory()
                st.session_state.user_question = ""
                st.session_state.clear_input = True
                st.session_state.acm_rolling_summary = ""
                st.session_state.acm_summarized_count = 0
                st.session_state.acm_memory_mode = "keep_all"
                st.session_state.last_source_docs = []
                st.session_state.last_source_question = ""
                st.session_state.last_response_time_seconds = 0.0
                st.session_state.last_source_count = 0
                st.session_state.last_feedback_id = ""
                st.session_state.last_feedback_question = ""
                st.session_state.last_feedback_answer = ""
                st.rerun()
        else:
            chosen = st.selectbox(
                "Lịch sử trò chuyện",
                list_chat_sessions,
                key="chat_session_free"
            )
            if chosen != st.session_state.history_choice:
                st.session_state.history_choice = chosen
                st.session_state.current_history = load_history(chosen)

                # Mỗi phiên hội thoại phải có trạng thái summary riêng.
                # Khi đổi phiên, reset và tạo lại từ lịch sử của phiên đó.
                st.session_state.acm_rolling_summary = ""
                st.session_state.acm_summarized_count = 0
                st.session_state.acm_memory_mode = "keep_all"

                st.session_state.last_source_docs = []
                st.session_state.last_source_question = ""
                st.session_state.last_response_time_seconds = 0.0
                st.session_state.last_source_count = 0
                st.session_state.last_feedback_id = ""
                st.session_state.last_feedback_question = ""
                st.session_state.last_feedback_answer = ""

                st.rerun()

    # ── Quản lý Lịch sử trò chuyện ──────────────────────────────────────────
    if "current_history" not in st.session_state:
        st.session_state.current_history = load_history(st.session_state.history_choice)

    history = st.session_state.current_history
    chat_container = st.container()

    # ── Hiển thị lịch sử chat trước ──────────────────────────────────────────
    with chat_container:
        for message in history.messages:
            st.chat_message(message["type"]).write(message["content"])

    # Sau st.rerun(), hiển thị lại nguồn của phản hồi gần nhất.
    if (
        st.session_state.rag_chat
        and st.session_state.last_source_docs
    ):
        render_source_documents(
            st.session_state.last_source_docs
        )

    if st.session_state.last_feedback_id:
        render_feedback_controls()

    # Ô nhập câu hỏi:
    # Chỉ gửi khi người dùng nhấn Enter hoặc nút "Gửi".
    # Không dùng on_change vì callback có thể chạy khi ô nhập mất focus.
    with st.form(
        "chat_input_form",
        clear_on_submit=True,
    ):
        typed_question = st.text_input(
            "Mẹ muốn hỏi điều gì?",
            key="user_input_widget",
            placeholder="Nhập câu hỏi rồi nhấn Enter để gửi...",
        )

        submitted = st.form_submit_button(
            "Gửi",
            use_container_width=True,
        )

    if submitted:
        normalized_question = str(
            typed_question or ""
        ).strip()

        if normalized_question:
            st.session_state.send_input = True
            st.session_state.user_question = normalized_question

    if st.session_state.clear_input:
        st.session_state.clear_input = False

    # ── Xử lý khi user gửi câu hỏi ──────────────────────────────────────────
    if st.session_state.send_input:
        user_text = st.session_state.user_question.strip()

        # Chỉ xóa trạng thái sau khi người dùng đã chủ động gửi.
        st.session_state.user_question = ""
        st.session_state.send_input = False
        st.session_state.locked_session = True

        if not user_text:
            st.rerun()
            return

        # Lịch sử an toàn được chuyển cho ACM. ACM phân tích tối đa 20 tin
        # nhắn, sau đó giữ nguyên hoặc tóm tắt tùy tổng độ dài trước khi
        # gửi sang Query Rewriting.
        # =========================================================
        # LỌC CÁC LƯỢT BỊ CHẶN KHỎI NGỮ CẢNH REWRITER
        # =========================================================
        filtered_history = []

        for msg in history.messages:
            msg_type = msg.get("type", "")
            content = str(msg.get("content", ""))

            # Nếu phản hồi AI là cảnh báo hoặc từ chối:
            # - không đưa phản hồi này vào ngữ cảnh;
            # - xóa luôn câu hỏi người dùng ngay trước đó.
            if (
                msg_type == "ai"
                and is_blocked_or_safety_response(content)
            ):
                if (
                    filtered_history
                    and filtered_history[-1].get("type") == "human"
                ):
                    filtered_history.pop()

                print(
                    "🛡️ [HISTORY FILTER] "
                    "Đã loại một lượt bị chặn khỏi ngữ cảnh Rewriter."
                )
                continue

            filtered_history.append(msg)


        # Chuẩn bị lịch sử bằng Adaptive Memory.
        (
            chat_history_messages,
            history_stats,
            updated_summary,
            updated_count,
        ) = build_adaptive_history(
            filtered_history=filtered_history,
            previous_summary=(
                st.session_state.acm_rolling_summary
            ),
            summarized_count=(
                st.session_state.acm_summarized_count
            ),
        )

        # Lưu trạng thái mới để dùng cho lượt sau.
        st.session_state.acm_rolling_summary = updated_summary
        st.session_state.acm_summarized_count = updated_count
        st.session_state.acm_memory_mode = history_stats["mode"]


        print("\n📚 [DEBUG ADAPTIVE MEMORY]")
        print(
            f"Chế độ: {history_stats['mode']} | "
            f"Tin nhắn: {history_stats['input_messages']} "
            f"-> {history_stats['output_messages']} | "
            f"Ký tự: {history_stats['total_chars_before']} "
            f"-> {history_stats['total_chars_after']} | "
            f"Đã tóm tắt: "
            f"{history_stats.get('summarized_count_before', 0)} "
            f"-> {history_stats.get('summarized_count_after', 0)} | "
            f"Chờ xử lý: {history_stats.get('pending_messages', 0)} tin"
        )

        for index, message in enumerate(
            chat_history_messages,
            start=1
        ):
            print(
                f"{index}. "
                f"{message.__class__.__name__}: "
                f"{message.content[:150]}"
            )

        print("------------------------------------\n")

        with chat_container:
            st.chat_message('human').write(user_text)

        with chat_container:
            with st.spinner("🤱 MomCare đang phân tích..."):
                request_started_at = time.perf_counter()

                try:
                    if st.session_state.rag_chat:
                        rag_chain = llm_chain.load_rag_chain_with_sources(
                            number_of_documents=DEFAULT_TOP_K,
                            temperature=temperature
                        )
                        result = rag_chain.invoke({"question": user_text, "history": chat_history_messages})
                        print("\n========== DEBUG RAG ==========")
                        print("Result keys:", result.keys())

                        source_docs = result.get("docs", [])

                        print("Number docs:", len(source_docs))

                        for i, d in enumerate(source_docs):
                            print(
                                i,
                                d.metadata.get("source"),
                                d.metadata.get("page"),
                                d.metadata.get("chunk_id")
                            )
                        print("==============================\n")
                        response = result.get(
                            "answer",
                            "⚠️ Hệ thống không tạo được câu trả lời.",
                        )
                        source_docs = result.get("docs", [])

                        # ===================================================
                        # LƯU NGUỒN ĐỂ HIỂN THỊ SAU st.rerun()
                        # ===================================================

                        st.session_state.last_source_docs = [
                            {
                                "page_content": str(doc.page_content),
                                "metadata": dict(doc.metadata),
                            }
                            for doc in source_docs
                        ]

                        st.session_state.last_source_question = user_text
                    else:
                        normal_chain = llm_chain.load_normal_chain(
                            temperature=temperature
                        )

                        response = normal_chain.invoke(
                            {
                                "question": user_text,
                                "history": chat_history_messages,
                            }
                        )

                        source_docs = []

                        st.session_state.last_source_docs = []
                        st.session_state.last_source_question = ""

                    if response:
                        st.chat_message('ai').write(response)
                    else:
                        response = "⚠️ Hệ thống AI đang gặp sự cố kết nối. Vui lòng thử lại sau."
                        st.chat_message('ai').write(response)

                except Exception as e:
                    import traceback

                    st.error(f"Lỗi: {e}")
                    print(traceback.format_exc())

                    response = (
                        "⚠️ Hệ thống đang gặp sự cố kỹ thuật. "
                        "Vui lòng thử lại sau."
                    )
                    source_docs = []
                    st.session_state.last_source_docs = []
                    st.session_state.last_source_question = ""

                response_time_seconds = (
                    time.perf_counter()
                    - request_started_at
                )

                # Lưu dữ liệu vận hành và định danh phản hồi gần nhất.
                st.session_state.last_response_time_seconds = (
                    response_time_seconds
                )
                st.session_state.last_source_count = len(
                    source_docs
                )
                st.session_state.last_feedback_question = user_text
                st.session_state.last_feedback_answer = response
                st.session_state.last_feedback_id = (
                    _make_feedback_id(
                        user_text,
                        response,
                    )
                )

                # Luôn lưu lượt hội thoại để giao diện hiển thị đầy đủ.
                history.add_a_conversation(
                    user_text,
                    response
                )

                if st.session_state.rag_chat:
                    st.session_state.last_source_docs = [
                        {
                            "page_content": str(
                                getattr(
                                    doc,
                                    "page_content",
                                    "",
                                )
                            ),
                            "metadata": dict(
                                getattr(
                                    doc,
                                    "metadata",
                                    {},
                                )
                                or {}
                            ),
                        }
                        for doc in source_docs
                    ]
                else:
                    st.session_state.last_source_docs = []

                st.rerun()
                    

# ════════════════════════════════════════
def _show_analysis_result(reason, reason_vi, confidence, acoustic_desc, source_type, REASON_TO_QUERY_MAP):
    if reason != "none" and confidence >= 0.015:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Nguồn âm thanh", source_type)
        with col2: st.metric("⚠️ Nguyên nhân", reason_vi)
        with col3: st.metric("🔊 Phát hiện tiếng khóc", f"{confidence*100:.1f}%")

        if acoustic_desc:
            with st.expander("📊 Xem chi tiết phân tích âm thanh", expanded=False):
                st.markdown(acoustic_desc)

        st.divider()
        rag_query = REASON_TO_QUERY_MAP.get(reason, REASON_TO_QUERY_MAP["unknown"])

        if reason == "unknown":
            st.subheader("🤖 Lời khuyên từ MomCare - Kiểm tra theo thứ tự ưu tiên")
            st.warning("⚠️ **Hệ thống phát hiện tiếng khóc nhưng chưa đủ dữ liệu để xác định nguyên nhân chính xác.**\n\nMẹ hãy kiểm tra theo thứ tự ưu tiên sau (mỗi bước 2-3 phút):")
            checklist = [
                ("🥛 1. KIỂM TRA ĐÓI", "Đưa ngón tay lên môi bé, nếu bé quay đầu tìm ti → Bé đói", "hunger"),
                ("🧷 2. KIỂM TRA TẢ", "Mở tã xem có ướt/đầy không", "discomfort"),
                ("🌡️ 3. KIỂM TRA NHIỆT ĐỘ", "Chạm tay vào gáy bé - ướt mồ hôi = quá nóng", "temperature"),
                ("😴 4. KIỂM TRA BUỒN NGỦ", "Mắt lờ đờ, ngáp → Bé buồn ngủ", "fatigue"),
                ("😰 5. KIỂM TRA ĐAU", "Sờ toàn thân, chỗ nào khóc to hơn → đau ở đó", "pain"),
            ]
            for title, desc, reason_key in checklist:
                col_check, col_text = st.columns([0.05, 0.95])
                with col_check:
                    st.checkbox("✓", key=f"check_{reason_key}", label_visibility="hidden")
                with col_text:
                    st.markdown(f"**{title}**\n> {desc}")
            st.info("💡 Nhấp vào checkbox khi đã kiểm tra xong từng bước.")
        else:
            st.subheader(f"🤖 Lời khuyên từ MomCare - Xử lý: {reason_vi}")

        with st.spinner("🤱 MomCare đang tìm kiếm tài liệu y khoa phù hợp..."):
            try:
                rag_chain = llm_chain.load_rag_chain_with_sources(
                    number_of_documents=DEFAULT_TOP_K,
                    temperature=DEFAULT_TEMPERATURE
                )
                result = rag_chain.invoke({
                    "question": user_text,
                    "history": chat_history_messages,
                    "previous_turn_blocked": (
                        st.session_state.previous_turn_blocked
                    ),
                })
                st.markdown(result["answer"])
                with st.expander("📎 Xem nguồn tài liệu y khoa", expanded=False):
                    if result["docs"]:
                        st.success(f"✅ Tìm thấy {len(result['docs'])} nguồn tài liệu")
                        for i, doc in enumerate(result["docs"]):
                            st.info(f"**📄 {i+1}:** `{doc.metadata.get('source', 'N/A')}`\n\n{doc.page_content[:300]}...")
                    else:
                        st.warning("❌ Không tìm thấy nguồn tài liệu phù hợp.")
            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")

    elif reason == "none":
        msg = reason_vi if reason_vi not in ["❌ KHÔNG PHÁT HIỆN KHÓC", "❌ LỖI PHÂN TÍCH"] \
              else "Hệ thống không phát hiện tiếng khóc rõ ràng. Vui lòng ghi âm lại khi bé đang khóc."
        st.warning(f"⚠️ {msg}")

# ════════════════════════════════════════
def Audio_Analysis():
    from streamlit_mic_recorder import mic_recorder
    import audio_utils

    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-size: 50px;">🎤</span>
        <h1 style="color: #ff4757; margin-top: -10px;">Phân tích tiếng khóc</h1>
        <p style="color: #636e72;">Hệ thống sẽ xác định <b>NGUYÊN NHÂN</b> bé khóc và đưa ra tư vấn cụ thể</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("👉 Nhấn **Bắt đầu ghi âm**, khi bé khóc thì nhấn **Dừng ghi âm**. Hệ thống sẽ tự động phân tích nguyên nhân.")

    REASON_TO_QUERY_MAP = {
        "hunger":      "Dấu hiệu trẻ sơ sinh đói, cách cho bú đúng kỹ thuật, lượng sữa cần thiết theo tuổi",
        "pain":        "Trẻ sơ sinh khóc do đau, nguyên nhân đau bụng kolik, khi nào cần đưa bé đi viện gấp",
        "fatigue":     "Cách dỗ trẻ sơ sinh buồn ngủ, kỹ thuật ru ngủ, giúp bé ngủ ngon, dấu hiệu bé buồn ngủ cần dỗ ngủ ngay",
        "discomfort":  "Cách thay tã đúng cách cho trẻ sơ sinh, dấu hiệu tã ướt cần thay, hăm tã ở trẻ sơ sinh",
        "temperature": "Nhiệt độ phòng lý tưởng cho trẻ sơ sinh, dấu hiệu bé quá nóng quá lạnh, cách mặc đồ theo thời tiết",
        "unknown":     "Trẻ sơ sinh khóc không rõ nguyên nhân, cách kiểm tra bé đói tã ướt buồn ngủ đau, cách dỗ bé nín khóc",
    }

    audio_bytes = mic_recorder(
        start_prompt="⏺ Bắt đầu ghi âm",
        stop_prompt="⏹ Dừng ghi âm",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    current_audio_bytes = audio_bytes['bytes'] if audio_bytes else None
    last_audio_bytes = st.session_state.get("last_audio_bytes")

    if current_audio_bytes and current_audio_bytes != last_audio_bytes:
        st.session_state.last_audio_bytes = current_audio_bytes
        st.success("✅ Đã ghi âm xong! Đang phân tích nguyên nhân...")
        with st.spinner("🧠 Đang phân tích dải tần số và chẩn đoán nguyên nhân khóc..."):
            reason, reason_vi, confidence, acoustic_desc = audio_utils.analyze_baby_cry(current_audio_bytes)
        _show_analysis_result(reason, reason_vi, confidence, acoustic_desc, "Ghi âm trực tiếp", REASON_TO_QUERY_MAP)

    elif not audio_bytes:
        st.markdown("---")
        uploaded_audio = st.file_uploader("HOẶC Tải file âm thanh lên (.wav, .mp3)", type=["wav", "mp3"], key="audio_uploader")

        if uploaded_audio is not None:
            audio_data = uploaded_audio.read()
            st.success("✅ Đã tải file lên thành công!")
            with st.spinner("🧠 Đang phân tích dải tần số và chẩn đoán nguyên nhân khóc..."):
                reason, reason_vi, confidence, acoustic_desc = audio_utils.analyze_baby_cry(audio_data)
            _show_analysis_result(reason, reason_vi, confidence, acoustic_desc, "Tải file lên", REASON_TO_QUERY_MAP)

# ════════════════════════════════════════════════════════════════════════════
def Databases():
    st.title('📑 Quản lý Kho Kiến thức')
    with st.sidebar:
        if st.button("🔄 Cập nhật VectorDB"):
            with st.spinner("Đang xử lý..."):
                create_vectordb_with_file()
                st.success("Cập nhật thành công!")
        data_type_option = st.selectbox("Loại dữ liệu", ["Tài liệu (PDF, Word, CSV)", "Lịch sử Chat"])

    if data_type_option == "Tài liệu (PDF, Word, CSV)":
        document_add = st.file_uploader("Thêm tài liệu", type=["pdf", "docx", "csv"])
        if document_add:
            if st.button("Tải lên"):
                if document_add.name.endswith(".pdf"):
                    path = os.path.join(db_config["pdf_path"], document_add.name)
                elif document_add.name.endswith(".docx"):
                    path = os.path.join(db_config["word_path"], document_add.name)
                else:
                    path = os.path.join(db_config.get("csv_path", "data_store/csv"), document_add.name)

                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(document_add.getbuffer())
                st.success(f"Đã lưu: {document_add.name}")

        st.divider()
        st.subheader("Danh sách tài liệu")
        
        all_docs = get_list_documents()
        excel_path = db_config.get("excel_path", "data_store/excel")
        excel_docs = []
        if os.path.exists(excel_path):
            excel_docs = [f for f in os.listdir(excel_path) if f.endswith('.xlsx') and not f.startswith('~$')]
        
        all_docs = all_docs + excel_docs
        selected_doc = st.selectbox("Chọn tài liệu", all_docs)
        if selected_doc:
            col1, col2, col3 = st.columns(3)
            if col1.button("👁️ Chi tiết"):
                _, text = get_details(selected_doc)
                st.text_area("Nội dung:", text, height=300)
            if col3.button("🗑️ Xóa"):
                if delete_document(selected_doc):
                    st.success("Đã xóa.")
                    st.rerun()

# ── Router ───────────────────────────────────────────────────────────────────
if selected == "Chatbot":
    Chatbot()
else:
    Databases()