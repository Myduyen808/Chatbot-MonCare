import os
import yaml
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load cấu hình đường dẫn từ file db_config.yml
with open("db_config.yml", "r", encoding="utf-8") as f:
    db_config = yaml.safe_load(f)

# Sửa lỗi cú pháp: dùng = thay vì :
pdf_path = db_config.get("pdf_path", "data_store/pdf")
docx_path = db_config.get("word_path", "data_store/word") 
excel_path = db_config.get("excel_path", "data_store/excel")
database_path = db_config.get("database_path", "data_store/vector_db/faiss") # Sửa ở đây

def count_files(folder_path, extension):
    """Hàm đếm số lượng file theo đuôi mở rộng trong một thư mục"""
    if not os.path.exists(folder_path):
        return 0
    
    count = 0
    for _, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extension):
                if not file.startswith("~$"):
                    count += 1
    return count

# 2. Đếm số lượng từng loại file
pdf_count = count_files(pdf_path, ".pdf")
docx_count = count_files(docx_path, ".docx")
excel_count = count_files(excel_path, ".xlsx")
total_files = pdf_count + docx_count + excel_count

# 3. ĐẾM SỐ LƯỢNG CHUNKS TỪ FAISS DB
chunk_count = 0
if os.path.exists(database_path):
    try:
        print("Đang load Embedding model để đọc DB...")
        # Load model (giống trong vectordb.py)
        with open("model_config.yml", "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        embeddings = HuggingFaceEmbeddings(model_name=model_config["embedding_path"])
        
        print("Đang đọc FAISS index...")
        # Load FAISS DB
        db = FAISS.load_local(database_path, embeddings, allow_dangerous_deserialization=True)
        
        # Lấy tổng số vector (chunks) lưu trong DB
        chunk_count = db.index.ntotal
    except Exception as e:
        chunk_count = -1 # Đánh dấu là bị lỗi
        print(f"Lỗi khi đọc FAISS DB: {e}")
else:
    print(f"Không tìm thấy thư mục DB tại: {database_path}")

# 4. In ra màn hình theo định dạng
print("\n" + "=" * 35)
print("     THỐNG KÊ DATABASE RAG")
print("=" * 35)
print(f"  PDF       : {pdf_count:>3} files")
print(f"  DOCX      : {docx_count:>3} files")
print(f"  XLSX      : {excel_count:>3} files")
print("-" * 35)
print(f"  TOTAL FILE: {total_files:>3} files")
print(f"  TOTAL DB  : {chunk_count:>3} chunks")
print("=" * 35)