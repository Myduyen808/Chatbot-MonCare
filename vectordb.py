# ====== Fix for Windows ======
import sys, os, yaml , re
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
    for folder_key, ext in [("word_path", ".docx"), ("pdf_path", ".pdf")]:
        path = db_config.get(folder_key, "")
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(ext) and not file.startswith("~$"): documents.append(file)
    csv_path = db_config.get("csv_path", "data_store/csv")
    if os.path.exists(csv_path):
        for root, dirs, files in os.walk(csv_path):
            for file in files:
                if file.endswith(".csv"): documents.append(file)
    return documents

def get_details(filename):
    text, filepath = "", ""
    if filename.endswith(".docx"):
        filepath = os.path.join(db_config["word_path"], filename); doc = Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith(".pdf"):
        filepath = os.path.join(db_config["pdf_path"], filename); reader = PdfReader(filepath)
        for page in reader.pages: text += (page.extract_text() or "")
    elif filename.endswith(".csv"):
        filepath = os.path.join(db_config.get("csv_path", "data_store/csv"), filename)
        with open(filepath, 'r', encoding='utf-8') as f: text = f.read()
    return {"Tên file": filename, "Đường dẫn": filepath, "Kích thước": os.path.getsize(filepath) if os.path.exists(filepath) else 0}, text

def get_document(filename):
    docs = []
    if filename.endswith(".docx"):
        doc = Document(os.path.join(db_config["word_path"], filename))
        docs.append(LC_Document(page_content="\n".join([p.text for p in doc.paragraphs if p.text.strip()]), metadata={"source": filename, "file_type": "docx"}))
    elif filename.endswith(".pdf"):
        docs = PyPDFLoader(os.path.join(db_config["pdf_path"], filename)).load()
        for d in docs: 
            d.page_content = clean_pdf_text(d.page_content) # Làm sạch rác ngay khi load
            d.metadata["file_type"] = "pdf"
    elif filename.endswith(".csv"):
        docs = CSVLoader(os.path.join(db_config.get("csv_path", "data_store/csv"), filename), encoding='utf-8').load()
        for d in docs: d.metadata["file_type"] = "csv"
    return docs

def delete_document(filename):
    for path in [db_config["word_path"], db_config["pdf_path"], db_config.get("csv_path", "data_store/csv")]:
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

    documents = pdf_documents + word_documents + csv_documents
    if not documents: print("Không tìm thấy tài liệu!"); return

    final_clean_documents = []
    for doc in documents:
        cleaned_text = clean_web_boilerplate(doc.page_content)
        if len(cleaned_text) > 50: final_clean_documents.append(LC_Document(page_content=cleaned_text, metadata=doc.metadata))
            
    # Tăng size và overlap để không bị cắt xẻ các bảng y khoa
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,      
        chunk_overlap=200     
    ).split_documents(final_clean_documents)

    # Clean noise VÀ lưu text đã clean
    cleaned_chunks = []
    for c in chunks:
        cleaned_text = clean_chunk_text(c.page_content)
        if len(cleaned_text.strip()) >= 100:
            cleaned_chunks.append(LC_Document(
                page_content=cleaned_text,
                metadata=c.metadata
            ))
    print(f"Chunks sau lọc: {len(cleaned_chunks)}")

    db = FAISS.from_documents(documents=cleaned_chunks, embedding=load_embedding())
    db.save_local(db_path)
    print(f"Đã tạo FAISS DB với {len(cleaned_chunks)} đoạn (Đã gắn thẻ file_type).")

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

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,         
            "lambda_mult": 0.6     
        }
    )

    results = retriever.invoke(question)

    # ── THÊM ĐOẠN NÀY: keyword overlap filter ──
    question_words = set(question.lower().split())
    filtered = [
        doc for doc in results
        if len(question_words & set(doc.page_content.lower().split())) >= 1
    ]
    # fallback nếu không có doc nào vượt ngưỡng
    if not filtered:
        filtered = results
    # ── KẾT THÚC ĐOẠN THÊM ──

    all_results = []
    seen_contents = set()

    for doc in results:
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