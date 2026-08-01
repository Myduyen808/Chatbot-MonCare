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
DEFAULT_TEMPERATURE = 0.2

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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    def update_send_input_state():
        if st.session_state["user_input_widget"].strip():
            st.session_state.send_input = True
            st.session_state.user_question = st.session_state["user_input_widget"].strip()

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
        temperature = st.slider(
            "Độ sáng tạo (Temperature)",
            min_value=0.0,
            max_value=0.5,
            value=DEFAULT_TEMPERATURE,
            step=0.1
        )
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

    # Ô nhập câu hỏi
    st.text_input(
        "Mẹ muốn hỏi điều gì?",
        key="user_input_widget",
        on_change=update_send_input_state,
        value="" if st.session_state.clear_input else st.session_state.user_question
    )

    if st.session_state.clear_input:
        st.session_state.clear_input = False

    # ── Xử lý khi user gửi câu hỏi ──────────────────────────────────────────
    if st.session_state.send_input:
        user_text = st.session_state.user_question.strip()
        
        # Giải phóng biến lưu trữ câu hỏi ngay để tránh kẹt bộ nhớ vòng lặp sau
        st.session_state.user_question = ""
        st.session_state.send_input = False
        st.session_state.locked_session = True
        st.session_state.clear_input = True

        if not user_text:
            st.rerun()
            return

        # Trích xuất 6 tin nhắn gần nhất để làm ngữ cảnh hội thoại, mỗi tin
        # nhắn dài hơn 200 ký tự sẽ được nén qua summarize_history_message()
        # để hạn chế Context Bleeding và tràn token khi gửi tới tầng Rewriter.
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


        # Chỉ lấy 6 tin nhắn an toàn gần nhất
        chat_history_messages = []

        for msg in filtered_history[-6:]:
            content = str(msg.get("content", ""))

            if len(content) > 200:
                content = llm_chain.summarize_history_message(content)

            if msg.get("type") == "human":
                chat_history_messages.append(
                    HumanMessage(content=content)
                )
            else:
                chat_history_messages.append(
                    AIMessage(content=content)
                )


        # Debug: kiểm tra lịch sử thực sự gửi sang Rewriter
        print("\n📚 [DEBUG HISTORY SENT TO REWRITER]")
        print(f"Số tin nhắn: {len(chat_history_messages)}")

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
                try:
                    if st.session_state.rag_chat:
                        rag_chain = llm_chain.load_rag_chain_with_sources(
                            number_of_documents=DEFAULT_TOP_K,
                            temperature=temperature
                        )
                        result = rag_chain.invoke({"question": user_text, "history": chat_history_messages})
                        response = result["answer"]
                        source_docs = result["docs"]
                    else:
                        normal_chain = llm_chain.load_normal_chain(temperature=temperature)
                        response = normal_chain.invoke({"question": user_text, "history": chat_history_messages})
                        source_docs = []

                    if response:
                        st.chat_message('ai').write(response)
                    else:
                        response = "⚠️ Hệ thống AI đang gặp sự cố kết nối. Vui lòng thử lại sau."
                        st.chat_message('ai').write(response)

                    if st.session_state.rag_chat and source_docs:
                        with st.expander("📎 Xem nguồn tài liệu", expanded=False):
                            st.success(f"✅ Tìm thấy {len(source_docs)} nguồn tài liệu")
                            for i, doc in enumerate(source_docs, start=1):
                                source_name = doc.metadata.get(
                                    "source",
                                    "Không rõ"
                                )
                                page = doc.metadata.get(
                                    "page_display",
                                    doc.metadata.get("page", "Không xác định")
                                )
                                chunk_id = doc.metadata.get(
                                    "chunk_id",
                                    "Không xác định"
                                )
                                file_type = doc.metadata.get(
                                    "file_type",
                                    "Không xác định"
                                )

                                st.info(
                                    f"**📄 Nguồn {i}:** `{source_name}`  \n"
                                    f"**Loại:** `{file_type}` | "
                                    f"**Trang:** `{page}` | "
                                    f"**Chunk:** `{chunk_id}`  \n\n"
                                    f"{doc.page_content[:500]}..."
                                )
                except Exception as e:
                    import traceback
                    st.error(f"Lỗi: {e}")
                    print(traceback.format_exc())
                    response = "Xin lỗi, tôi đang gặp sự cố."

        # Luôn lưu mọi lượt để giao diện giữ nguyên cuộc trò chuyện
        history.add_a_conversation(user_text, response)

        if is_blocked_or_safety_response(response):
            st.session_state.previous_turn_blocked = True

            print(
                "🛡️ [HISTORY] Đã lưu để hiển thị, "
                "nhưng sẽ loại khỏi ngữ cảnh Rewriter."
            )
        else:
            st.session_state.previous_turn_blocked = False
            print("💾 [HISTORY] Đã lưu lượt hội thoại hợp lệ.")
                    

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