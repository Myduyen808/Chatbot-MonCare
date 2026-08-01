# ====== Fix for Windows ======
import sys, os, yaml , re
import pandas as pd
if sys.platform == "win32":
    try: import mock
    except ImportError: os.system("pip install mock"); import mock
    sys.modules["pwd"] = mock.Mock()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from docx import Document
from langchain_core.documents import Document as LC_Document
from pypdf import PdfReader

# Thêm cache ở đầu file, sau phần import
_vector_db_cache = None

with open("db_config.yml", "r", encoding="utf-8") as f: db_config = yaml.safe_load(f)
with open("model_config.yml", "r", encoding="utf-8") as f: model_config = yaml.safe_load(f)

# Thêm sau dòng _vector_db_cache = None
_embedding_model = None  # Cache model ở đây

def clean_pdf_text(text):
    """Xóa rác lỗi font thường gặp khi trích xuất PDF"""
    # 1. Xóa mã rác bullet point bị lỗi font (uf075, uf0b7...)
    text = re.sub(r'uf0[0-9a-fA-F]{2}', '', text)
    # 2. Thay chữ 'n' bị lỗi xuống dòng bằng khoảng trắng
    text = re.sub(r'\bn\b', ' ', text)
    # 3. Xóa các ký tự điều khiển ẩn
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 4. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_embedding():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=model_config["embedding_path"])
    return _embedding_model

def clean_web_boilerplate(text):
    noise_keywords = [
        "Top of Form", "Bottom of Form", "ĐỂ LẠI THÔNG TIN TƯ VẤN", "Họ và tên*", 
        "Số điện thoại*", "Tôi đã đọc và đồng ý với Chính sách bảo vệ dữ liệu cá nhân", 
        "Cập nhật:", "Chia sẻ", "Tải và đặt lịch khám tự động trên ứng dụng MyVinmec",
        "Câu hỏi thảo luận", "anh/chị thấy gì", "nội dung tranh lật", 
        "hướng dẫn cách sử dụng tranh lật", "các kỹ năng cần thiết", "lời tranh",
        "để nấu một bữa", "Chatbot này tư vấn được những vấn đề gì"
    ]
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not any(noise in line for noise in noise_keywords)]
    clean_text = "\n".join(cleaned_lines).strip()
    paragraphs = clean_text.split('\n\n')
    unique_paragraphs, seen = [], set()
    for p in paragraphs:
        normalized_p = " ".join(p.split())
        if normalized_p not in seen and len(normalized_p) > 30: 
            seen.add(normalized_p); unique_paragraphs.append(p)
    return "\n\n".join(unique_paragraphs).strip()

def get_list_documents():
    documents = []

    file_configs = [
        ("word_path", ".docx"),
        ("pdf_path", ".pdf"),
        ("csv_path", ".csv"),
        ("excel_path", ".xlsx"),
    ]

    for folder_key, ext in file_configs:
        path = db_config.get(folder_key, "")

        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(ext) and not file.startswith("~$"):
                        documents.append(file)

    return documents

def get_details(filename):
    text, filepath = "", ""

    if filename.endswith(".docx"):
        filepath = os.path.join(db_config["word_path"], filename)
        doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif filename.endswith(".pdf"):
        filepath = os.path.join(db_config["pdf_path"], filename)
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += (page.extract_text() or "")

    elif filename.endswith(".csv"):
        filepath = os.path.join(
            db_config.get("csv_path", "data_store/csv"),
            filename
        )
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    elif filename.endswith(".xlsx"):
        filepath = os.path.join(
            db_config.get("excel_path", "data_store/excel"),
            filename
        )
        df = pd.read_excel(filepath)
        text = df.to_string(index=False)

    return {
        "Tên file": filename,
        "Đường dẫn": filepath,
        "Kích thước": os.path.getsize(filepath) if os.path.exists(filepath) else 0
    }, text

def get_document(filename):
    docs = []

    if filename.endswith(".docx"):
        doc = Document(os.path.join(db_config["word_path"], filename))
        docs.append(
            LC_Document(
                page_content="\n".join(
                    [p.text for p in doc.paragraphs if p.text.strip()]
                ),
                metadata={
                    "source": filename,
                    "file_type": "docx"
                }
            )
        )

    elif filename.endswith(".pdf"):
        docs = PyPDFLoader(
            os.path.join(db_config["pdf_path"], filename)
        ).load()

        for d in docs:
            d.page_content = clean_pdf_text(d.page_content)
            d.metadata["file_type"] = "pdf"

    elif filename.endswith(".csv"):
        docs = CSVLoader(
            os.path.join(
                db_config.get("csv_path", "data_store/csv"),
                filename
            ),
            encoding="utf-8"
        ).load()

        for d in docs:
            d.metadata["file_type"] = "csv"

    elif filename.endswith(".xlsx"):
        filepath = os.path.join(
            db_config.get("excel_path", "data_store/excel"),
            filename
        )

        docs = load_special_excel(filepath)

    return docs

def delete_document(filename):
    for path in [db_config["word_path"], db_config["pdf_path"], db_config.get("csv_path", "data_store/csv"),db_config.get("excel_path", "data_store/excel")]:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath): os.remove(filepath); return True
    return False

class MyPyWordLoader:
    def __init__(self, word_path=db_config["word_path"]): self.word_path = word_path
    def load(self):
        docx_documents = []
        for root, dirs, files in os.walk(self.word_path):
            for file in files:
                if file.endswith(".docx") and not file.startswith("~$"):
                    document = Document(os.path.join(root, file))
                    text = "\n".join([p.text for p in document.paragraphs if p.text.strip()])
                    if text.strip(): docx_documents.append(LC_Document(page_content=text, metadata={"source": file, "file_type": "docx"}))
        return docx_documents


def load_special_excel(file_path):
    """Nạp file Excel thành Document"""
    documents = []

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print(f"\nĐang đọc: {file_path}")
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))

    # chuẩn hóa tên cột
    df.columns = [str(c).strip().lower() for c in df.columns]

    TOPIC_MAP = {0: "Body Part", 1: "Disease", 2: "Drug", 3: "Medicine"}

    for _, row in df.iterrows():

        # ── Bộ đề sản khoa (medical_exam) ──────────────────────────
        if {"medical_topic", "question", "options", "answer"}.issubset(df.columns):

            content = (
                f"Chuyên khoa: {row.get('medical_topic', '')}. "
                f"Câu hỏi: {row.get('question', '')}. "
                f"Các lựa chọn: {row.get('options', '')}. "
                f"Đáp án: {row.get('answer', '')}."
            )

            metadata = {
                "source": os.path.basename(file_path),
                "file_type": "medical_exam"
            }

        # ── ViMedAQA format — CÓ CỘT context (Bo_De_Me_Va_Be.xlsx) ─
        # Ưu tiên trước FAQ vì có context đầy đủ hơn nhiều
        elif {"question", "answer", "context"}.issubset(df.columns):

            topic_id   = row.get("topic", "")
            topic_name = TOPIC_MAP.get(int(topic_id), "Y khoa") if str(topic_id).isdigit() else str(topic_id)
            title      = str(row.get("title", "")).strip()
            keyword    = str(row.get("keyword", "")).strip()
            question   = str(row.get("question", "")).strip()
            answer     = str(row.get("answer", "")).strip()
            context    = str(row.get("context", "")).strip()

            # Bỏ qua hàng rỗng
            if not question or not context:
                continue

            content = (
                f"Chủ đề: {topic_name}. "
                + (f"Tiêu đề: {title}. " if title and title != "nan" else "")
                + (f"Từ khoá: {keyword}. " if keyword and keyword != "nan" else "")
                + f"Câu hỏi: {question}. "
                + f"Trả lời: {answer}. "
                + f"Ngữ cảnh: {context}"
            )

            metadata = {
                "source"   : os.path.basename(file_path),
                "file_type": "vimedaqa",
                "topic"    : topic_name,
                "title"    : title,
            }

        # ── FAQ thông thường — chỉ có question + answer ─────────────
        elif {"question", "answer"}.issubset(df.columns):

            content = (
                f"Câu hỏi: {row.get('question', '')}. "
                f"Trả lời: {row.get('answer', '')}."
            )

            metadata = {
                "source": os.path.basename(file_path),
                "file_type": "faq"
            }

        else:
            print(
                f"⚠ Không nhận diện được cấu trúc file: "
                f"{os.path.basename(file_path)}"
            )
            continue

        documents.append(
            LC_Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents

# Thêm hàm này vào vectordb.py
def is_data_driven_chunk(text: str) -> bool:
    """
    Phát hiện đoạn văn chứa nhiều dữ liệu định lượng (bảng, thống kê).
    Nếu là bảng dữ liệu -> KHÔNG cắt nhỏ, giữ nguyên 1 chunk.
    """
    # Đếm số dòng chứa số liệu (mg, ml, tháng, tuổi, %)
    lines = text.split('\n')
    data_lines = 0
    for line in lines:
        if re.search(r'(\d+\s*(mg|ml|g|%|tháng|tuổi|ngày|lần|tuần|kcal|kg))', line.lower()):
            data_lines += 1
            
    # Nếu >30% dòng chứa số liệu -> đây là bảng/chuỗi thống kê -> Không cắt
    if len(lines) > 2 and (data_lines / len(lines)) > 0.3:
        return True
    return False


def create_vectordb_with_file(pdf_path=db_config["pdf_path"], word_path=db_config["word_path"], csv_path=db_config.get("csv_path", "data_store/csv"), chunk_size=db_config["database_config"]["chunk_size"], chunk_overlap=db_config["database_config"]["chunk_overlap"], db_path=db_config["database_path"]):
    os.makedirs(pdf_path, exist_ok=True); os.makedirs(word_path, exist_ok=True); os.makedirs(csv_path, exist_ok=True)
    
    pdf_documents = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader).load()
    for doc in pdf_documents:
        doc.page_content = clean_pdf_text(doc.page_content) # <--- Làm sạch rác PDF
        doc.metadata["file_type"] = "pdf"
    
    word_documents = MyPyWordLoader(word_path).load()
    
    csv_documents_raw = DirectoryLoader(csv_path, glob="*.csv", loader_cls=CSVLoader, loader_kwargs={'encoding': 'utf-8'}).load()
    csv_documents = []
    for doc in csv_documents_raw:
        lines = [line.replace('noidung_doan:', '').strip() for line in doc.page_content.split('\n') if not (line.startswith('tieude:') or line.startswith('nguon:') or line.strip() == '')]
        clean_text = "\n".join(lines).strip()
        if len(clean_text) > 50: csv_documents.append(LC_Document(page_content=clean_text, metadata={**doc.metadata, "file_type": "csv"}))

    # Load Excel đặc biệt
    excel_path = db_config.get("excel_path", "data_store/excel")
    
    special_docs = []

    print("========== EXCEL DEBUG ==========")

    excel_path = db_config.get("excel_path", "data_store/excel")
    print("Excel path:", excel_path)
    print("Exists:", os.path.exists(excel_path))

    if os.path.exists(excel_path):

        for root, dirs, files in os.walk(excel_path):

            for file in files:

                if file.endswith(".xlsx") and not file.startswith("~$"):

                    full_path = os.path.join(root, file)

                    special_docs.extend(
                        load_special_excel(full_path)
                    )

                    print(f"Đã nạp file Excel: {file}")
                    print("Found Excel:", file)
            
    print("Số document Excel:", len(special_docs))

    documents = (
    pdf_documents
    + word_documents
    + csv_documents
    + special_docs
    )
    if not documents: print("Không tìm thấy tài liệu!"); return

    final_clean_documents = []
    for doc in documents:
        cleaned_text = clean_web_boilerplate(doc.page_content)
        if len(cleaned_text) > 50: final_clean_documents.append(LC_Document(page_content=cleaned_text, metadata=doc.metadata))
            
    # Sử dụng biến chunk_size và chunk_overlap được truyền từ cấu hình/YAML
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    final_chunks = []

    for doc in final_clean_documents:

        # FAQ, ViMedAQA hoặc bộ đề y khoa giữ nguyên 1 document (không split)
        if doc.metadata.get("file_type") in ["medical_exam", "faq", "vimedaqa"]:
            cleaned_text = clean_chunk_text(doc.page_content)
            if len(cleaned_text.strip()) >= 50:
                final_chunks.append(
                    LC_Document(page_content=cleaned_text, metadata=doc.metadata)
                )

        # CÁI MỚI: Phát hiện bảng số liệu -> Giữ nguyên không cắt
        elif is_data_driven_chunk(doc.page_content):
            cleaned_text = clean_chunk_text(doc.page_content)
            if len(cleaned_text.strip()) >= 80:
                # Gắn thêm flag để Reranker ưu tiên sau này
                doc.metadata["chunk_type"] = "data_table"
                final_chunks.append(
                    LC_Document(page_content=cleaned_text, metadata=doc.metadata)
                )

        # Các tài liệu khác mới chia chunk
        else:
            chunks = splitter.split_documents([doc])
            for c in chunks:
                cleaned_text = clean_chunk_text(c.page_content)
                if len(cleaned_text.strip()) >= 50:
                    c.metadata["chunk_type"] = "normal_text"
                    final_chunks.append(
                        LC_Document(page_content=cleaned_text, metadata=c.metadata)
                    )
                    
    # Gắn định danh truy vết cho từng chunk.
    for chunk_index, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = chunk_index

        # PyPDFLoader thường đánh số trang từ 0.
        # Tạo page_display để giao diện hiển thị từ trang 1.
        raw_page = chunk.metadata.get("page")

        if isinstance(raw_page, int):
            chunk.metadata["page_display"] = raw_page + 1

    print(f"Chunks sau lọc: {len(final_chunks)}")

    db = FAISS.from_documents(
        documents=final_chunks,
        embedding=load_embedding()
    )

    db.save_local(db_path)

    print(f"Đã tạo FAISS DB với {len(final_chunks)} đoạn (Đã gắn thẻ file_type).")

def load_vector_db(db_path=db_config["database_path"]):
    global _vector_db_cache
    if _vector_db_cache is None:
        _vector_db_cache = FAISS.load_local(
            db_path, load_embedding(), 
            allow_dangerous_deserialization=True
        )
    return _vector_db_cache

def detect_query_priority(question):
    q = question.lower()
    mom_baby_keywords = ["núm vú", "cho con bú", "sữa mẹ", "sau sinh", "tắc sữa", "tắc tia sữa", "viêm tuyến vú", "mách sữa", "áp-xe vú", "massage vú", "hút sữa", "bầu vú", "rốn", "tưa miệng", "rôm sảy", "khóc dạ đề", "ăn dặm"]
    for kw in mom_baby_keywords:
        if kw in q: return ["docx", "pdf"]
    return ["pdf", "docx"]

# ===== SMART RETRIEVE ĐÃ SỬA =====
def smart_retrieve(question, llm, k=5, score_threshold=100.0):
    db = load_vector_db()

    # ── Tăng fetch_k để có pool lớn hơn, đặc biệt cho câu hỏi định lượng ──
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 30,         # tăng từ 20 → 30
            "lambda_mult": 0.7     # tăng từ 0.6 → 0.7 (ưu tiên relevance hơn diversity)
        }
    )

    results = retriever.invoke(question)

    # ── Keyword overlap filter — dùng filtered thay vì bỏ qua nó ──
    question_words = set(question.lower().split())
    filtered = [
        doc for doc in results
        if len(question_words & set(doc.page_content.lower().split())) >= 1
    ]
    # fallback nếu không có doc nào vượt ngưỡng
    if not filtered:
        filtered = results

    all_results = []
    seen_contents = set()

    # ── Dùng filtered (đã fix bug: trước đây loop trên results, không phải filtered) ──
    for doc in filtered:
        content = str(doc.page_content)
        content_hash = hash(content[:200])
        
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            all_results.append(doc)

    return all_results[:k]

# Thêm hàm này vào vectordb.py
def clean_chunk_text(text: str) -> str:
    """Xóa dòng rác nhưng GIỮ LẠI nội dung hữu ích trong cùng chunk"""
    noise_line_patterns = [
        "câu hỏi thảo luận",
        "anh/chị thấy gì",
        "nội dung tranh lật",
        "hướng dẫn cách sử dụng tranh lật",
        "các kỹ năng cần thiết",
        "lời tranh",
        "để nấu một bữa",   
    ]
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line_lower = line.lower().strip()
        if any(p in line_lower for p in noise_line_patterns):
            continue
        if line_lower and line_lower[0].isdigit() and '.' in line_lower[:3]:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()

def is_meaningful_chunk(text: str) -> bool:
    """Chỉ bỏ chunk nếu sau khi xóa noise vẫn còn quá ngắn"""
    cleaned = clean_chunk_text(text)
    return len(cleaned.strip()) >= 100

def get_retriever(k=5):
    return load_vector_db().as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    print("Đang nạp dữ liệu..."); create_vectordb_with_file(); print("Xong!")