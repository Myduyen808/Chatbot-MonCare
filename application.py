from unittest import result

import csv
import io
import sqlite3
import hmac
import secrets
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

from history_handle import CustomHistory
import llm_chain
from vectordb import (
    get_list_documents,
    get_document,
    delete_document,
    get_details,
    create_vectordb_with_file,
    reset_vector_db_cache,
)

DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.0

FEEDBACK_DB_PATH = Path("runtime_logs") / "user_feedback.db"

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
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] {background-color: #ffffff; box-shadow: 2px 0 10px rgba(0,0,0,0.05);}
.stButton > button {background-color: #ff6b81; color: white; border-radius: 20px; border: none; font-weight: bold;}
.stButton > button:hover {background-color: #ff4757; transform: translateY(-2px);}
</style>
""", unsafe_allow_html=True)

def _get_operator_password() -> str:
    """Đọc mật khẩu vận hành từ .env hoặc Streamlit secrets."""
    env_password = os.getenv(
        "MOMCARE_OPERATOR_PASSWORD",
        "",
    ).strip()

    if env_password:
        return env_password

    try:
        auth_config = st.secrets.get("auth", {})

        return str(
            auth_config.get(
                "operator_password",
                "",
            )
        ).strip()

    except Exception:
        # Không có secrets.toml thì ứng dụng vẫn chạy
        # ở chế độ dành cho người dùng.
        return ""


def _is_operator() -> bool:
    """Kiểm tra quyền của phiên hiện tại."""
    return bool(
        st.session_state.get(
            "operator_authenticated",
            False,
        )
    )


def _render_operator_login():
    """Hiển thị đăng nhập và đăng xuất cho người vận hành."""
    st.markdown("---")

    if _is_operator():
        st.success(
            "🔓 Đã đăng nhập: Người vận hành"
        )

        if st.button(
            "Đăng xuất",
            key="operator_logout_button",
            use_container_width=True,
        ):
            st.session_state.operator_authenticated = False
            st.rerun()

        return

    with st.expander(
        "🔐 Đăng nhập vận hành",
        expanded=False,
    ):
        operator_password = st.text_input(
            "Mật khẩu",
            type="password",
            key="operator_password_input",
            placeholder="Nhập mật khẩu vận hành",
        )

        if st.button(
            "Đăng nhập",
            key="operator_login_button",
            use_container_width=True,
        ):
            configured_password = (
                _get_operator_password()
            )

            if not configured_password:
                st.error(
                    "Chưa cấu hình "
                    "MOMCARE_OPERATOR_PASSWORD "
                    "trong tệp .env."
                )

            elif hmac.compare_digest(
                str(operator_password),
                configured_password,
            ):
                st.session_state.operator_authenticated = True
                st.rerun()

            else:
                st.error("Mật khẩu không đúng.")

# ── Menu chính ──────────────────────────────────────────────────────────────

if "operator_authenticated" not in st.session_state:
    st.session_state.operator_authenticated = False

# Người dùng thông thường chỉ được thấy Chatbot.
menu_options = ["Chatbot"]
menu_icons = ["chat"]

# Chỉ người vận hành mới thấy Quản lý Dữ liệu.
if _is_operator():
    menu_options.append("Quản lý Dữ liệu")
    menu_icons.append("database")

with st.sidebar:
    selected = option_menu(
        "Menu Chính",
        menu_options,
        icons=menu_icons,
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {
                "font-family": "sans-serif"
            },
            "nav-link-selected": {
                "background-color": "#ff4b4b"
            },
        },
    )

    _render_operator_login()

def clear_cache():
    """Xóa toàn bộ cache retrieval sau khi VectorDB đổi."""

    st.cache_resource.clear()

    reset_vector_db_cache()

    if hasattr(
        llm_chain,
        "reset_retrieval_caches",
    ):
        llm_chain.reset_retrieval_caches()

def rag_click():
    st.session_state.rag_chat = True

@st.cache_resource
def load_chain(rag_chat, number_of_documents):
    if rag_chat:
        return llm_chain.load_rag_chain(number_of_documents)
    return llm_chain.load_normal_chain()

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

def _feedback_connect():
    """Mở SQLite database lưu feedback."""
    FEEDBACK_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        FEEDBACK_DB_PATH,
        timeout=10,
    )

    # Cho phép nhiều phiên đọc/ghi ổn định hơn.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            rating TEXT NOT NULL
                CHECK (rating IN ('helpful', 'not_helpful')),
            source_count INTEGER NOT NULL DEFAULT 0,
            response_time_seconds REAL NOT NULL DEFAULT 0
        )
        """
    )

    return conn


def _save_user_feedback(
    feedback_id: str,
    rating: str,
    source_count: int,
    response_time_seconds: float,
):
    """Lưu feedback bằng SQLite transaction."""

    if rating not in {
        "helpful",
        "not_helpful",
    }:
        return

    conn = _feedback_connect()

    try:
        with conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    feedback_id,
                    timestamp,
                    rating,
                    source_count,
                    response_time_seconds
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(feedback_id) DO UPDATE SET
                    rating = excluded.rating,
                    source_count = excluded.source_count,
                    response_time_seconds =
                        excluded.response_time_seconds
                """,
                (
                    feedback_id,
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    rating,
                    int(source_count),
                    round(
                        float(response_time_seconds),
                        3,
                    ),
                ),
            )

    finally:
        conn.close()


def _read_user_feedback():
    """Đọc feedback từ SQLite."""

    conn = _feedback_connect()
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT
                timestamp,
                feedback_id,
                rating,
                source_count,
                response_time_seconds
            FROM feedback
            ORDER BY timestamp DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def _feedback_to_csv(rows):
    """Tạo CSV khi operator yêu cầu tải xuống."""

    output = io.StringIO()

    fieldnames = [
        "timestamp",
        "feedback_id",
        "rating",
        "source_count",
        "response_time_seconds",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(
            {
                field: row.get(field, "")
                for field in fieldnames
            }
        )

    return (
        "\ufeff" + output.getvalue()
    ).encode("utf-8")


def _render_feedback_management():
    """Giao diện thống kê feedback cho operator."""

    st.subheader(
        "📊 Thống kê phản hồi người dùng"
    )

    if st.button(
        "🔄 Làm mới thống kê",
        key="refresh_feedback_statistics",
    ):
        st.rerun()

    feedback_rows = _read_user_feedback()

    if not feedback_rows:
        st.info(
            "Chưa có phản hồi nào được ghi nhận."
        )
        return

    total_count = len(feedback_rows)

    helpful_count = sum(
        row.get("rating") == "helpful"
        for row in feedback_rows
    )

    not_helpful_count = sum(
        row.get("rating") == "not_helpful"
        for row in feedback_rows
    )

    helpful_rate = (
        helpful_count / total_count * 100
        if total_count
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tổng đánh giá",
        total_count,
    )

    col2.metric(
        "👍 Hữu ích",
        helpful_count,
    )

    col3.metric(
        "👎 Chưa hữu ích",
        not_helpful_count,
    )

    col4.metric(
        "Tỷ lệ hữu ích",
        f"{helpful_rate:.1f}%",
    )

    rating_filter = st.selectbox(
        "Lọc theo đánh giá",
        [
            "Tất cả",
            "Hữu ích",
            "Chưa hữu ích",
        ],
        key="operator_feedback_filter",
    )

    rating_map = {
        "Hữu ích": "helpful",
        "Chưa hữu ích": "not_helpful",
    }

    selected_rating = rating_map.get(
        rating_filter
    )

    filtered_rows = [
        row
        for row in feedback_rows
        if (
            selected_rating is None
            or row.get("rating")
            == selected_rating
        )
    ]

    st.dataframe(
        filtered_rows,
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = _feedback_to_csv(
        filtered_rows
    )

    st.download_button(
        "⬇️ Tải dữ liệu phản hồi CSV",
        data=csv_bytes,
        file_name="user_feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )

def _make_feedback_id() -> str:
    """Tạo mã ngẫu nhiên, không suy ra từ nội dung hội thoại."""
    return secrets.token_hex(8)


def _empty_chat_record(number: int) -> dict:
    """Tạo một cuộc trò chuyện chỉ tồn tại trong session_state."""
    return {
        "title": f"Cuộc trò chuyện {number}",
        "history": CustomHistory(),
        "previous_turn_blocked": False,
        "acm_rolling_summary": "",
        "acm_summarized_count": 0,
        "acm_memory_mode": "keep_all",
        "last_source_docs": [],
        "last_source_question": "",
        "last_response_time_seconds": 0.0,
        "last_source_count": 0,
        "last_feedback_id": "",
        "feedback_by_id": {},
    }


def _ensure_temporary_chat_state() -> None:
    """Khởi tạo và sửa trạng thái chat tạm sau khi Streamlit hot-reload."""
    sessions = st.session_state.get("chat_sessions")

    if not isinstance(sessions, dict) or not sessions:
        first_session_id = secrets.token_hex(6)
        sessions = {
            first_session_id: _empty_chat_record(1)
        }
        st.session_state.chat_sessions = sessions
        st.session_state.chat_session_counter = 1
        st.session_state.loaded_chat_session_id = first_session_id
        st.session_state.chat_session_selector = first_session_id
        _load_temporary_chat(first_session_id)
        return

    # Phiên Streamlit tạo bởi bản mã cũ có thể đã có chat_sessions
    # nhưng chưa có biến đếm. Dùng số phiên hiện có làm giá trị bắt đầu.
    if "chat_session_counter" not in st.session_state:
        st.session_state.chat_session_counter = max(
            1,
            len(sessions),
        )

    # Bổ sung các trường còn thiếu nếu cấu trúc phiên được tạo bởi
    # phiên bản mã trước đó.
    for position, (session_id, record) in enumerate(
        list(sessions.items()),
        start=1,
    ):
        if not isinstance(record, dict):
            sessions[session_id] = _empty_chat_record(position)
            continue

        default_record = _empty_chat_record(position)
        for key, value in default_record.items():
            record.setdefault(key, value)

    loaded_id = st.session_state.get("loaded_chat_session_id")
    selected_id = st.session_state.get("chat_session_selector")

    if loaded_id not in sessions:
        loaded_id = (
            selected_id
            if selected_id in sessions
            else next(iter(sessions))
        )
        st.session_state.loaded_chat_session_id = loaded_id
        st.session_state.chat_session_selector = loaded_id
        _load_temporary_chat(loaded_id)
    elif selected_id not in sessions:
        st.session_state.chat_session_selector = loaded_id


def _chat_title(session_id: str) -> str:
    """Lấy tên hiển thị của một cuộc trò chuyện tạm thời."""
    record = st.session_state.get("chat_sessions", {}).get(
        session_id,
        {},
    )
    history = record.get("history")
    if history is not None and history.messages:
        return history.history_name or record.get(
            "title",
            "Cuộc trò chuyện",
        )
    return record.get("title", "Cuộc trò chuyện")


def _capture_active_chat() -> None:
    """Chụp trạng thái ACM hiện tại trước khi chuyển cuộc trò chuyện."""
    sessions = st.session_state.get("chat_sessions", {})
    session_id = st.session_state.get("loaded_chat_session_id")
    if session_id not in sessions:
        return

    record = sessions[session_id]
    record["history"] = st.session_state.current_history

    for key in (
        "previous_turn_blocked",
        "acm_rolling_summary",
        "acm_summarized_count",
        "acm_memory_mode",
        "last_source_docs",
        "last_source_question",
        "last_response_time_seconds",
        "last_source_count",
        "last_feedback_id",
        "feedback_by_id",
    ):
        record[key] = st.session_state.get(key)


def _load_temporary_chat(session_id: str) -> None:
    """Nạp một cuộc trò chuyện từ session_state, không đọc tệp trên đĩa."""
    record = st.session_state.chat_sessions[session_id]
    history = record["history"]

    st.session_state.current_history = history
    st.session_state.loaded_chat_session_id = session_id
    st.session_state.locked_session = bool(history.messages)
    st.session_state.user_question = ""
    st.session_state.send_input = False
    st.session_state.clear_input = True

    for key in (
        "previous_turn_blocked",
        "acm_rolling_summary",
        "acm_summarized_count",
        "acm_memory_mode",
        "last_source_docs",
        "last_source_question",
        "last_response_time_seconds",
        "last_source_count",
        "last_feedback_id",
        "feedback_by_id",
    ):
        st.session_state[key] = record[key]


def _on_chat_session_change() -> None:
    """Chuyển sang cuộc trò chuyện được chọn trong sidebar."""
    _ensure_temporary_chat_state()
    selected_id = st.session_state.get("chat_session_selector")
    loaded_id = st.session_state.get("loaded_chat_session_id")
    if not selected_id or selected_id == loaded_id:
        return
    _capture_active_chat()
    _load_temporary_chat(selected_id)


def _create_temporary_chat() -> None:
    """Tạo cuộc trò chuyện mới trong phiên trình duyệt hiện tại."""
    _ensure_temporary_chat_state()
    _capture_active_chat()
    st.session_state.chat_session_counter = (
        int(st.session_state.get("chat_session_counter", 0))
        + 1
    )
    session_id = secrets.token_hex(6)
    st.session_state.chat_sessions[session_id] = _empty_chat_record(
        st.session_state.chat_session_counter
    )
    st.session_state.chat_session_selector = session_id
    _load_temporary_chat(session_id)


def _delete_temporary_chat() -> None:
    """Xóa riêng cuộc trò chuyện đang chọn và toàn bộ ngữ cảnh ACM của nó."""
    _ensure_temporary_chat_state()
    sessions = st.session_state.chat_sessions
    session_id = st.session_state.loaded_chat_session_id
    sessions.pop(session_id, None)

    if not sessions:
        st.session_state.chat_session_counter = (
            int(st.session_state.get("chat_session_counter", 0))
            + 1
        )
        next_id = secrets.token_hex(6)
        sessions[next_id] = _empty_chat_record(
            st.session_state.chat_session_counter
        )
    else:
        next_id = next(iter(sessions))

    st.session_state.chat_session_selector = next_id
    _load_temporary_chat(next_id)


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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Nhiều cuộc trò chuyện được giữ tạm trong phiên trình duyệt.
    # Không có nội dung nào trong cấu trúc này được ghi xuống đĩa.
    _ensure_temporary_chat_state()

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

        session_ids = list(st.session_state.chat_sessions)
        st.selectbox(
            "Cuộc trò chuyện",
            session_ids,
            key="chat_session_selector",
            format_func=_chat_title,
            on_change=_on_chat_session_change,
        )

        col_new_chat, col_delete_chat = st.columns(2)
        col_new_chat.button(
            "➕ Tạo mới",
            use_container_width=True,
            on_click=_create_temporary_chat,
        )
        col_delete_chat.button(
            "🗑️ Xóa",
            use_container_width=True,
            on_click=_delete_temporary_chat,
        )

    # ── Quản lý Lịch sử trò chuyện ──────────────────────────────────────────
    if "current_history" not in st.session_state:
        st.session_state.current_history = CustomHistory()

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
        border=False,
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
                st.session_state.last_feedback_id = (
                    _make_feedback_id()
                )

                # Chỉ giữ hội thoại trong session_state. CustomHistory không
                # ghi nội dung xuống đĩa, nhưng ACM vẫn đọc được messages.
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
    # Kiểm tra quyền lần thứ hai.
    # Việc ẩn menu không được xem là đủ để bảo vệ chức năng.
    if not _is_operator():
        st.error(
            "Bạn cần đăng nhập với quyền "
            "Người vận hành để truy cập chức năng này."
        )
        return

    st.title("📑 Quản lý Kho Kiến thức")

    with st.sidebar:
        if st.button("🔄 Cập nhật VectorDB"):
            with st.spinner(
                "Đang xây dựng lại VectorDB..."
            ):
                create_vectordb_with_file()
                clear_cache()
                st.success("Cập nhật thành công!")

        data_type_option = st.selectbox(
            "Loại dữ liệu",
            [
                "Tài liệu (PDF, DOCX, XLSX, CSV)",
                "Phản hồi người dùng",
            ],
        )

    # =====================================================
    # QUẢN LÝ TÀI LIỆU
    # =====================================================
    if (
        data_type_option
        == "Tài liệu (PDF, DOCX, XLSX, CSV)"
    ):
        document_add = st.file_uploader(
            "Thêm tài liệu",
            type=[
                "pdf",
                "docx",
                "xlsx",
                "csv",
            ],
        )

        if document_add:
            if st.button("Tải lên"):
                # Loại phần đường dẫn không an toàn
                # khỏi tên tệp do người dùng tải lên.
                safe_filename = Path(
                    document_add.name
                ).name

                extension = Path(
                    safe_filename
                ).suffix.lower()

                if extension == ".pdf":
                    data_folder = db_config[
                        "pdf_path"
                    ]

                elif extension == ".docx":
                    data_folder = db_config[
                        "word_path"
                    ]

                elif extension == ".xlsx":
                    data_folder = db_config.get(
                        "excel_path",
                        "data_store/excel",
                    )

                else:
                    data_folder = db_config.get(
                        "csv_path",
                        "data_store/csv",
                    )

                path = os.path.join(
                    data_folder,
                    safe_filename,
                )

                os.makedirs(
                    os.path.dirname(path),
                    exist_ok=True,
                )

                # Nếu đang thay file cùng tên thì backup bản cũ.
                old_file_data = None

                if os.path.exists(path):
                    with open(path, "rb") as file:
                        old_file_data = file.read()

                with open(path, "wb") as file:
                    file.write(
                        document_add.getbuffer()
                    )

                try:
                    with st.spinner(
                        "Đang đồng bộ tài liệu với VectorDB..."
                    ):
                        create_vectordb_with_file()
                        clear_cache()

                    st.success(
                        f"Đã thêm và đồng bộ: {safe_filename}"
                    )

                except Exception as error:

                    # Nếu rebuild thất bại:
                    # - file mới hoàn toàn → xóa
                    # - file thay thế → khôi phục bản cũ.
                    if old_file_data is None:

                        if os.path.exists(path):
                            os.remove(path)

                    else:
                        with open(path, "wb") as file:
                            file.write(old_file_data)

                    st.error(
                        "Cập nhật VectorDB thất bại. "
                        "Tệp nguồn đã được hoàn nguyên."
                    )

                    st.exception(error)

        st.divider()
        st.subheader("Danh sách tài liệu")

        # set() tránh hiển thị trùng tên.
        all_docs = sorted(
            set(get_list_documents())
        )

        if not all_docs:
            st.info(
                "Kho dữ liệu chưa có tài liệu."
            )
            return

        selected_doc = st.selectbox(
            "Chọn tài liệu",
            all_docs,
        )

        if selected_doc:
            col_detail, col_confirm, col_delete = (
                st.columns(3)
            )

            if col_detail.button("👁️ Chi tiết"):
                _, text = get_details(
                    selected_doc
                )

                st.text_area(
                    "Nội dung:",
                    text,
                    height=300,
                )

            confirm_delete = col_confirm.checkbox(
                "Xác nhận xóa",
                key=f"confirm_delete_{selected_doc}",
            )

            if col_delete.button(
                "🗑️ Xóa",
                disabled=not confirm_delete,
            ):
                if delete_document(selected_doc):
                    st.success("Đã xóa tài liệu.")

                    st.warning(
                        "Cần cập nhật lại VectorDB "
                        "để loại tài liệu khỏi "
                        "chỉ mục truy xuất."
                    )

                    st.rerun()

    # =====================================================
    # XEM PHẢN HỒI NGƯỜI DÙNG
    # =====================================================
    if (
        data_type_option
        == "Tài liệu (PDF, DOCX, XLSX, CSV)"
    ):
        # Code quản lý tài liệu
        pass

    elif (
        data_type_option
        == "Phản hồi người dùng"
    ):
        _render_feedback_management()

# ── Router ───────────────────────────────────────────────────────────────────
if (
    selected == "Quản lý Dữ liệu"
    and _is_operator()
):
    Databases()
else:
    Chatbot()
