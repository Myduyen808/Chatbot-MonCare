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

# ẢNH
# def process_image_for_groq(uploaded_file):
#     if uploaded_file is None:
#         return None
#     compressed_buffer = compress_image_for_groq(uploaded_file)
#     image_bytes = compressed_buffer.read()
#     base64_image = base64.b64encode(image_bytes).decode("utf-8").replace("\n", "")
#     return [
#         {
#             "type": "text",
#             "text": "Hãy phân tích ảnh y khoa này và trả về JSON ngắn gọn theo mẫu: {\"vị_trí\": \"\", \"mô_tả_ngắn\": \"\", \"từ_khóa_truy_vấn\": [\"\"]}"
#         },
#         {
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#         }
#     ]

# ── Menu chính (ngoài function, chạy 1 lần mỗi render) ──────────────────────
with st.sidebar:
    selected = option_menu(
        "Menu Chính",
        ["Chatbot", "Phân tích tiếng khóc", "Quản lý Dữ liệu"],
        icons=['chat', 'mic', 'database'],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={"container": {"font-family": "sans-serif"}, "nav-link-selected": {"background-color": "#ff4b4b"}}
    )

def clear_cache():
    st.cache_resource.clear()

def rag_click():
    st.session_state.rag_chat = True
    clear_cache()

@st.cache_resource
def load_chain():
    if st.session_state.get("rag_chat", False):
        return llm_chain.load_rag_chain(st.session_state.number_of_documents)
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
        "number_of_documents": 3,
        "rag_chat": True,
        # FIX: 2 biến quản lý session nằm TRONG function
        "history_choice": "New Session",
        "locked_session": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    def update_send_input_state():
        st.session_state.send_input = True
        st.session_state.user_question = st.session_state["user_input_widget"]

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-size: 50px;">🤱</span>
        <h1 style="color: #ff4757; margin-top: -10px;">MomCare</h1>
        <p style="color: #636e72; font-size: 16px; margin-top: -10px;">Trợ lý AI Chuyên gia Chăm sóc Mẹ và Bé</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt Hệ thống")
        temperature = st.slider("Độ sáng tạo (Temperature)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        rag_button = st.toggle("Bật Chế độ RAG", value=True)

        if rag_button:
            st.info("🟢 Đang dùng tài liệu.")
            number_of_documents = st.slider("Số đoạn tài liệu (K)", min_value=1, max_value=15, value=5)
            # if number_of_documents > 5:
            #     st.warning(f" K={number_of_documents}: Hệ thống sẽ tóm tắt {number_of_documents} đoạn — phản hồi chậm hơn bình thường.")
            st.session_state.number_of_documents = number_of_documents
            if not st.session_state.get("rag_chat", False):
                rag_click()
        else:
            st.warning("🟠 Đang dùng kiến thức chung.")
            st.session_state.rag_chat = False

        st.markdown("---")

        # FIX: Quản lý session đúng chỗ, KHÔNG còn selectbox trùng
        list_chat_sessions = ["New Session"] + get_list_names()

        if st.session_state.locked_session:
            # Đang trong cuộc trò chuyện → khoá selectbox, hiện nút tạo mới
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
                st.session_state.clear_input = True
                st.rerun()
        else:
            # Chưa gửi tin → cho phép chọn session tự do
            chosen = st.selectbox(
                "Lịch sử trò chuyện",
                list_chat_sessions,
                key="chat_session_free"
            )
            st.session_state.history_choice = chosen

    # ── Load history theo session đang chọn ─────────────────────────────────
    if "current_history" not in st.session_state:
        st.session_state.current_history = load_history(st.session_state.history_choice)

    if not st.session_state.locked_session:
        st.session_state.current_history = load_history(st.session_state.history_choice)

    history = st.session_state.current_history
    chat_container = st.container()

    uploaded_image = None

    user_question = st.text_input(
        "Mẹ muốn hỏi điều gì?",
        key="user_input_widget",
        on_change=update_send_input_state,
        value="" if st.session_state.clear_input else st.session_state.get("user_question", "")
    )

    if st.session_state.clear_input:
        st.session_state.clear_input = False

    # ── Hiển thị lịch sử chat ────────────────────────────────────────────────
    with chat_container:
        for message in history.messages:
            st.chat_message(message["type"]).write(message["content"])

    # ── Xử lý khi user gửi câu hỏi ──────────────────────────────────────────
    if st.session_state.send_input:
        if not st.session_state.user_question:
            st.warning("Vui lòng nhập câu hỏi!")
            st.session_state.send_input = False
            return

        # FIX: Lock session ngay khi user gửi tin đầu tiên
        st.session_state.locked_session = True
        st.session_state.send_input = False

        chat_history_messages = []
        for msg in history.messages:
            if msg["type"] == "human":
                chat_history_messages.append(HumanMessage(content=msg["content"]))
            else:
                chat_history_messages.append(AIMessage(content=msg["content"]))
        user_text = st.session_state.user_question if st.session_state.user_question else "Hãy quan sát hình ảnh này."

        with chat_container:
            st.chat_message('human').write(user_text)

        with chat_container:
            with st.spinner("🤱 MomCare đang phân tích..."):
                try:
                    if st.session_state.get("rag_chat", False):
                        query = user_text
                        rag_chain = llm_chain.load_rag_chain_with_sources(
                            number_of_documents=st.session_state.number_of_documents,
                            temperature=temperature
                        )
                        result = rag_chain.invoke({"question": query, "history": chat_history_messages})
                        response = result["answer"]
                        source_docs = result["docs"]

                    else:
                        normal_chain = llm_chain.load_normal_chain(temperature=temperature)
                        response = normal_chain.invoke({"question": user_text, "history": chat_history_messages})
                        source_docs = []

                    if response:
                        st.chat_message('ai').write(response)
                    else:
                        st.chat_message('ai').write("⚠️ Hệ thống AI đang gặp sự cố kết nối. Vui lòng thử lại sau.")

                    with st.expander("📎 Xem nguồn tài liệu", expanded=False):
                        if not source_docs:
                            st.warning("❌ Không tìm thấy nguồn tài liệu phù hợp.")
                        else:
                            st.success(f"✅ Tìm thấy {len(source_docs)} nguồn tài liệu")
                            for i, doc in enumerate(source_docs):
                                source_name = doc.metadata.get('source', 'Không rõ')
                                st.info(f"**📄 {i+1}:** `{source_name}`\n\n{doc.page_content[:500]}...")

                except Exception as e:
                    import traceback
                    st.error(f"Lỗi: {e}")
                    print(traceback.format_exc())
                    response = "Xin lỗi, tôi đang gặp sự cố."

        # Chỉ lưu vào lịch sử nếu nó là câu hỏi RAG hợp lệ, không lưu câu bị Guardrails chặn oan
        is_blocked_response = any(kw in response for kw in [
            "MomCare không thể hỗ trợ yêu cầu này", 
            "MomCare không thể tư vấn về các sản phẩm không rõ nguồn gốc",
            "DỪNG LẠI! Hành động này rất nguy hiểm",
            "MomCare không thể cung cấp thông tin kê đơn",
            "MomCare không thể tư vấn về liều lượng thuốc",
            "CẢNH BÁO: Tình trạng của mẹ cần được xử lý Y TẾ NGAY"
        ])
        
        if not is_blocked_response:
            history.add_a_conversation(user_text, response)
            
        st.session_state.clear_input = True

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

    # ═══════════════════════════════════════════════════════════════
    # MAP NGUYÊN NHÂN → QUERY RAG CỤ THỂ
    # Đây là KEY FIX: Thay vì query chung "bé đang khóc", 
    # ta query CỤ THỂ theo nguyên nhân đã xác định
    # ═══════════════════════════════════════════════════════════════
    REASON_TO_QUERY_MAP = {
"hunger":      "Dấu hiệu trẻ sơ sinh đói, cách cho bú đúng kỹ thuật, lượng sữa cần thiết theo tuổi",
"pain":        "Trẻ sơ sinh khóc do đau, nguyên nhân đau bụng kolik, khi nào cần đưa bé đi viện gấp",
"fatigue": "Cách dỗ trẻ sơ sinh buồn ngủ, kỹ thuật ru ngủ, giúp bé ngủ ngon, dấu hiệu bé buồn ngủ cần dỗ ngủ ngay",
"discomfort":  "Cách thay tã đúng cách cho trẻ sơ sinh, dấu hiệu tã ướt cần thay, hăm tã ở trẻ sơ sinh",
"temperature": "Nhiệt độ phòng lý tưởng cho trẻ sơ sinh, dấu hiệu bé quá nóng quá lạnh, cách mặc đồ theo thời tiết",
"unknown":     "Trẻ sơ sinh khóc không rõ nguyên nhân, cách kiểm tra bé đói tã ướt buồn ngủ đau, cách dỗ bé nín khóc",
    }

    audio_data = None
    source_type = ""

    # ==========================================
    # GHI ÂM TRỰC TIẾP & TỰ ĐỘNG PHÂN TÍCH
    # ==========================================
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
        audio_data = current_audio_bytes
        source_type = "Ghi âm trực tiếp"
        st.success("✅ Đã ghi âm xong! Đang phân tích nguyên nhân...")

        # ==================== PHÂN TÍCH ====================
        with st.spinner("🧠 Đang phân tích dải tần số và chẩn đoán nguyên nhân khóc..."):
            reason, reason_vi, confidence, acoustic_desc = audio_utils.analyze_baby_cry(audio_data)
        
        # ==================== XỬ LÝ KẾT QUẢ ====================
        if reason != "none" and confidence >= 0.015:
            
            # ── HIỂN THỊ KẾT QUẢ PHÂN TÍCH ──
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nguồn âm thanh", source_type)
            with col2:
                # ✅ FIX: HIỂN THỊ NGUYÊN NHÂN CỤ THỂ thay vì "ĐANG KHÓC"
                st.metric("⚠️ Nguyên nhân", reason_vi)
            with col3:
                st.metric("🔊 Phát hiện tiếng khóc", f"{confidence*100:.1f}%")
            
            # Hiển thị chi tiết phân tích (expandable)
            if acoustic_desc:
                with st.expander("📊 Xem chi tiết phân tích âm thanh (Pitch, Energy, Rhythm)", expanded=False):
                    st.markdown(acoustic_desc)
            
            st.divider()
            
            # ── LẤY QUERY RAG CỤ THỂ THEO NGUYÊN NHÂN ──
            rag_query = REASON_TO_QUERY_MAP.get(reason, REASON_TO_QUERY_MAP["unknown"])
            
            # Hiển thị tiêu đề phù hợp
            if reason == "unknown":
                st.subheader("🤖 Lời khuyên từ MomCare - Kiểm tra theo thứ tự ưu tiên")
                
                # Hiển thị thông báo rõ ràng hơn khi không xác định được
                st.warning("""⚠️ **Hệ thống phát hiện tiếng khóc nhưng chưa đủ dữ liệu để xác định nguyên nhân chính xác.**

            Mẹ hãy kiểm tra theo thứ tự ưu tiên sau (mỗi bước 2-3 phút):""")
                
                # Hiển thị checklist trực quan
                checklist = [
                    ("🥛 1. KIỂM TRA ĐÓI", "Đưa ngón tay lên môi bé, nếu bé quay đầu tìm ti hoặc mút tay → Bé đói", "hunger"),
                    ("🧷 2. KIỂM TRA TẢ", "Mở tã xem có ướt/đầy không, kiểm tra vùng da có bị hăm đỏ không", "discomfort"),
                    ("🌡️ 3. KIỂM TRA NHIỆT ĐỘ", "Chạm tay vào gáy bé - nếu ướt mồ hôi = quá nóng, nếu lạnh = cần ấm hơn", "temperature"),
                    ("😴 4. KIỂM TRA BUỒN NGỦ", "Nếu bé đã thức quá lâu so với giờ giấc bình thường, mắt lờ đờ, ngáp → Bé buồn ngủ", "fatigue"),
                    ("😰 5. KIỂM TRA ĐAU", "Nhẹ nhàng sờ toàn thân bé, nếu chạm vào đâu bé khóc to hơn → Có thể bị đau ở đó", "pain"),
                ]
                
                for title, desc, reason_key in checklist:
                    with st.container():
                        col_check, col_text = st.columns([0.05, 0.95])
                        with col_check:
                            st.checkbox("✓", key=f"check_{reason_key}", label_visibility="hidden")
                        with col_text:
                            st.markdown(f"**{title}**\n> {desc}")
                
                st.markdown("---")
                st.info("💡 **Mẹ nhấp vào checkbox khi đã kiểm tra xong từng bước** để nhớ xem đã thử cách nào chưa.")
            else:
                st.subheader(f"🤖 Lời khuyên từ MomCare - Xử lý: {reason_vi}")
            
            # ── GỌI RAG VỚI QUERY CỤ THỂ ──
            with st.spinner("🤱 MomCare đang tìm kiếm tài liệu y khoa phù hợp..."):
                try:
                    rag_chain = llm_chain.load_rag_chain_with_sources(number_of_documents=5, temperature=0.3)
                    result = rag_chain.invoke({"question": rag_query, "history": []})
                    
                    response = result["answer"]
                    source_docs = result["docs"]
                    
                    st.markdown(response)
                    
                    with st.expander("📎 Xem nguồn tài liệu y khoa", expanded=False):
                        if not source_docs:
                            st.warning("❌ Không tìm thấy nguồn tài liệu phù hợp.")
                        else:
                            st.success(f"✅ Tìm thấy {len(source_docs)} nguồn tài liệu")
                            for i, doc in enumerate(source_docs):
                                st.info(f"**📄 {i+1}:** `{doc.metadata.get('source', 'N/A')}`\n\n{doc.page_content[:300]}...")
                                
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}")
        
        elif reason == "none":
            st.warning("⚠️ Hệ thống không phát hiện tiếng khóc rõ ràng. Vui lòng ghi âm lại khi bé đang khóc hoặc thử tải file khác.")

    # ==========================================
    # CÁCH 2: TẢI FILE LÊN
    # ==========================================
    elif not audio_bytes:
        st.markdown("---")
        uploaded_audio = st.file_uploader("HOẶC Tải file âm thanh lên (.wav, .mp3)", type=["wav", "mp3"], key="audio_uploader")
        
        if uploaded_audio is not None:
            audio_data = uploaded_audio.read()
            source_type = "Tải file lên"
            st.success("✅ Đã tải file lên thành công!")
            
            # Phân tích luôn khi tải file lên
            with st.spinner("🧠 Đang phân tích dải tần số và chẩn đoán nguyên nhân khóc..."):
                reason, reason_vi, confidence, acoustic_desc = audio_utils.analyze_baby_cry(audio_data)
                
            if reason != "none" and confidence >= 0.015:
                # ── HIỂN THỊ KẾT QUẢ PHÂN TÍCH ──
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nguồn âm thanh", source_type)
                with col2:
                    # ✅ FIX: HIỂN THỊ NGUYÊN NHÂN CỤ THỂ
                    st.metric("⚠️ Nguyên nhân", reason_vi)
                with col3:
                    st.metric("🔊 Phát hiện tiếng khóc", f"{confidence*100:.1f}%")
                
                # Hiển thị chi tiết phân tích
                if acoustic_desc:
                    with st.expander("📊 Xem chi tiết phân tích âm thanh", expanded=False):
                        st.markdown(acoustic_desc)
                
                st.divider()
                
                # ── LẤY QUERY RAG CỤ THỂ ──
                rag_query = REASON_TO_QUERY_MAP.get(reason, REASON_TO_QUERY_MAP["unknown"])
                
                if reason == "unknown":
                    st.subheader("🤖 Lời khuyên từ MomCare - Kiểm tra theo thứ tự ưu tiên")
                    
                    # Hiển thị thông báo rõ ràng hơn khi không xác định được
                    st.warning("""⚠️ **Hệ thống phát hiện tiếng khóc nhưng chưa đủ dữ liệu để xác định nguyên nhân chính xác.**

                Mẹ hãy kiểm tra theo thứ tự ưu tiên sau (mỗi bước 2-3 phút):""")
                    
                    # Hiển thị checklist trực quan
                    checklist = [
                        ("🥛 1. KIỂM TRA ĐÓI", "Đưa ngón tay lên môi bé, nếu bé quay đầu tìm ti hoặc mút tay → Bé đói", "hunger"),
                        ("🧷 2. KIỂM TRA TẢ", "Mở tã xem có ướt/đầy không, kiểm tra vùng da có bị hăm đỏ không", "discomfort"),
                        ("🌡️ 3. KIỂM TRA NHIỆT ĐỘ", "Chạm tay vào gáy bé - nếu ướt mồ hôi = quá nóng, nếu lạnh = cần ấm hơn", "temperature"),
                        ("😴 4. KIỂM TRA BUỒN NGỦ", "Nếu bé đã thức quá lâu so với giờ giấc bình thường, mắt lờ đờ, ngáp → Bé buồn ngủ", "fatigue"),
                        ("😰 5. KIỂM TRA ĐAU", "Nhẹ nhàng sờ toàn thân bé, nếu chạm vào đâu bé khóc to hơn → Có thể bị đau ở đó", "pain"),
                    ]
                    
                    for title, desc, reason_key in checklist:
                        with st.container():
                            col_check, col_text = st.columns([0.05, 0.95])
                            with col_check:
                                st.checkbox("✓", key=f"check_{reason_key}", label_visibility="hidden")
                            with col_text:
                                st.markdown(f"**{title}**\n> {desc}")
                    
                    st.markdown("---")
                    st.info("💡 **Mẹ nhấp vào checkbox khi đã kiểm tra xong từng bước** để nhớ xem đã thử cách nào chưa.")
                else:
                    st.subheader(f"🤖 Lời khuyên từ MomCare - Xử lý: {reason_vi}")
                
                # ── GỌI RAG VỚI QUERY CỤ THỂ ──
                with st.spinner("🤱 MomCare đang tìm kiếm tài liệu y khoa phù hợp..."):
                    try:
                        rag_chain = llm_chain.load_rag_chain_with_sources(number_of_documents=5, temperature=0.3)
                        result = rag_chain.invoke({"question": rag_query, "history": []})
                        
                        response = result["answer"]
                        source_docs = result["docs"]
                        
                        st.markdown(response)
                        
                        with st.expander("📎 Xem nguồn tài liệu y khoa", expanded=False):
                            if not source_docs:
                                st.warning("❌ Không tìm thấy nguồn tài liệu phù hợp.")
                            else:
                                st.success(f"✅ Tìm thấy {len(source_docs)} nguồn tài liệu")
                                for i, doc in enumerate(source_docs):
                                    st.info(f"**📄 {i+1}:** `{doc.metadata.get('source', 'N/A')}`\n\n{doc.page_content[:300]}...")
                                    
                    except Exception as e:
                        st.error(f"Lỗi hệ thống: {e}")
            
            elif reason == "none":
                st.warning("⚠️ Hệ thống không phát hiện tiếng khóc rõ ràng. Vui lòng thử file khác.")


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
        
        # Lấy danh sách từ hàm gốc
        all_docs = get_list_documents()
        
        # Lấy riêng danh sách file Excel (thêm mới)
        excel_path = db_config.get("excel_path", "data_store/excel")
        excel_docs = []
        if os.path.exists(excel_path):
            excel_docs = [f for f in os.listdir(excel_path) if f.endswith('.xlsx') and not f.startswith('~$')]
        
        # Gộp lại thành 1 danh sách đầy đủ
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


# ── Xóa cache để load thư viện mới ──────────────────────────────────────────
st.cache_resource.clear()

# ── Router ───────────────────────────────────────────────────────────────────
if selected == "Chatbot":
    Chatbot()
elif selected == "Phân tích tiếng khóc":
    Audio_Analysis()
else:
    Databases()