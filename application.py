import streamlit as st
from streamlit_option_menu import option_menu
import yaml
import os
# import base64
# import cv2
# import numpy as np
# from PIL import Image
# import io
import asyncio

from dotenv import load_dotenv
load_dotenv()

from utils import typewriter_effect
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

# # ── Menu chính (ngoài function, chạy 1 lần mỗi render) ──────────────────────
with st.sidebar:
    selected = option_menu(
        "Menu Chính",
        ["Chatbot", "Quản lý Dữ liệu"],
        icons=['chat', 'database'],
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

        chat_history_messages = get_chat_messages(history)
        user_text = st.session_state.user_question if st.session_state.user_question else "Hãy quan sát hình ảnh này."

        with chat_container:
            # if uploaded_image:
            #     col_img, col_txt = st.columns([1, 3])
            #     with col_img:
            #         st.image(uploaded_image, width=100)
            #     with col_txt:
            #         st.markdown(f"**Mẹ bỉm sữa:** {user_text}")
            # else:
                st.chat_message('human').write(user_text)

        with chat_container:
            with st.spinner("🤱 MomCare đang phân tích..."):
                try:
                    if st.session_state.get("rag_chat", False):
                        query = user_text

                        # if uploaded_image:
                        #     img_pil = Image.open(uploaded_image)
                        #     features = {}
                        #     for name in llm_chain.create_momcare_kernels():
                        #         _, intensity = llm_chain.apply_convolution_to_image(img_pil, name)
                        #         features[name] = {'intensity': intensity}

                        #     top3 = sorted(features.items(), key=lambda x: x[1]['intensity'], reverse=True)[:3]
                        #     top_kernel_name = top3[0][0]
                        #     col1, col2, col3 = st.columns(3)
                        #     for i, (kname, _) in enumerate(top3):
                        #         fmap, _ = llm_chain.apply_convolution_to_image(img_pil, kname)
                        #         with [col1, col2, col3][i]:
                        #             st.image(fmap, caption=f"{kname}: {features[kname]['intensity']:.0f}", width=120)

                        #     vision_chain = llm_chain.describe_image_chain()
                        #     vision_raw = vision_chain.invoke({"input": llm_chain.process_image_momcare(uploaded_image)})
                        #     vision_data = llm_chain.parse_vision_output(vision_raw, features)

                        #     query = llm_chain.build_query_from_question({
                        #         "vision_summary": vision_data["vision_summary"],
                        #         "question": user_text,
                        #         "features": features
                        #     })
                            # st.success(f"Chẩn đoán: **{top_kernel_name.replace('_', ' ').title()}**")

                        rag_chain = llm_chain.load_rag_chain_with_sources(
                            number_of_documents=st.session_state.number_of_documents,
                            temperature=temperature
                        )
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = rag_chain.invoke({"question": query, "history": chat_history_messages})
                        loop.close()
                        response = result["answer"]
                        source_docs = result["docs"]

                    else:
                        normal_chain = llm_chain.load_normal_chain(temperature=temperature)
                        response = normal_chain.invoke({"question": user_text, "history": chat_history_messages})
                        source_docs = []

                    st.chat_message('ai').write(response)

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

        history.add_a_conversation(user_text, response)
        st.session_state.clear_input = True


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


# ── Router ───────────────────────────────────────────────────────────────────
if selected == "Chatbot":
    Chatbot()
else:
    Databases()