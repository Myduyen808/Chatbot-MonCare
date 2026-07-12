import os
# 🌟 THẦN CHÚ CHỐNG CRASH NGẦM TRÊN WINDOWS (PHẢI ĐẶT Ở ĐẦU FILE)
# Khóa số luồng xử lý đa nhân để triệt tiêu lỗi sập ngầm OpenMP khi nạp sbert và FAISS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from vectordb import create_vectordb_with_file, db_config

# 1. Định nghĩa thư mục chứa 3 file PDF COVID 
pdf_folder = "data_generalization"
os.makedirs(pdf_folder, exist_ok=True)

print(f"📌 Bước 1: Đảm bảo đã copy 3 file PDF dữ liệu COVID vào thư mục: '{pdf_folder}'.")

# Thay bằng:
os.makedirs("temp_covid/word_empty", exist_ok=True)
os.makedirs("temp_covid/csv_empty", exist_ok=True)
os.makedirs("temp_covid/excel_empty", exist_ok=True)

# 🌟 KHÓA ĐƯỜNG DẪN EXCEL: Ép cấu hình hệ thống trỏ vào folder trống 
db_config["excel_path"] = "temp_covid/excel_empty"

# 3. Tiến hành gọi hàm chính chủ để bóc tách text và build Vector DB mới tinh khiết
print("\n⏳ Hệ thống đang chạy Hybrid Indexing (FAISS + BM25) CHỈ cho dữ liệu COVID-19...")
try:
    create_vectordb_with_file(
        pdf_path=pdf_folder,
        word_path="temp_covid/word_empty",
        csv_path="temp_covid/csv_empty",
        chunk_size=db_config["database_config"]["chunk_size"],
        chunk_overlap=db_config["database_config"]["chunk_overlap"],
        db_path="faiss_index_covid"  # Tạo bộ não chuyên biệt riêng cho COVID
    )
    print("\n✅ THÀNH CÔNG! Thư mục bộ não mới 'faiss_index_covid' đã xuất hiện trong dự án.")
except Exception as e:
    print(f"❌ Lỗi trong quá trình nạp dữ liệu: {e}")