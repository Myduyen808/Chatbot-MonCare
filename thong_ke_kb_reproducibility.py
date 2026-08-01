"""
Script tính thống kê tái lập (reproducibility) cho kho tri thức MomCare:
- Token trung bình / chunk (dùng tiktoken cl100k_base để có con số "token" chuẩn,
  tương tự cách các paper/luận văn RAG khác báo cáo, không phụ thuộc vào
  tokenizer riêng của Groq).
- Ký tự trung bình / chunk (bonus, dễ đối chiếu chéo).
- Thời điểm cập nhật gần nhất của FAISS index (dựa theo thời gian sửa đổi
  file index.faiss trên đĩa).

CÁCH CHẠY:
1. Copy file này vào đúng thư mục project (cùng cấp với vectordb.py, model_config.yml).
2. Kích hoạt đúng môi trường: (mom_env) 
3. Nếu chưa có tiktoken: pip install tiktoken --break-system-packages
4. Chạy: python thong_ke_kb_reproducibility.py

Script sẽ in kết quả ra terminal VÀ lưu vào file thong_ke_kb_output.txt để bạn
chụp màn hình hoặc đính kèm luận văn giống Hình 3.2/3.3 (kb_stats_terminal.png).
"""

import os
import sys
import datetime
import statistics

# ── Cố gắng dùng tiktoken để đếm token chuẩn; nếu chưa cài, fallback sang đếm từ ──
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
    TOKEN_METHOD = "tiktoken (cl100k_base)"
except ImportError:
    def count_tokens(text: str) -> int:
        # Fallback thô: đếm theo từ (word count), chỉ dùng khi chưa cài tiktoken
        return len(text.split())
    TOKEN_METHOD = "word-split (CHƯA CHÍNH XÁC - hãy cài tiktoken để có số liệu đáng tin cậy hơn)"

def main():
    output_lines = []
    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    log("=" * 70)
    log("THỐNG KÊ TÁI LẬP KHO TRI THỨC MOMCARE")
    log("=" * 70)
    log(f"Thời điểm chạy script này: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Phương pháp đếm token   : {TOKEN_METHOD}")
    log("")

    # ── Bước 1: Nạp FAISS index bằng đúng hàm hệ thống đã dùng ──
    try:
        from vectordb import load_vector_db
    except ImportError:
        log("LỖI: Không tìm thấy vectordb.py trong thư mục hiện tại.")
        log("Hãy copy script này vào đúng thư mục chứa vectordb.py rồi chạy lại.")
        sys.exit(1)

    log("Đang nạp FAISS index (có thể mất vài giây)...")
    db = load_vector_db()
    log("Nạp thành công.\n")

    # ── Bước 2: Duyệt toàn bộ chunk trong docstore ──
    # LangChain FAISS lưu toàn bộ Document trong db.docstore._dict
    try:
        all_docs = list(db.docstore._dict.values())
    except AttributeError:
        log("LỖI: Không truy cập được db.docstore._dict — cấu trúc FAISS store")
        log("có thể khác phiên bản langchain đang cài. Hãy kiểm tra lại thủ công.")
        sys.exit(1)

    total_chunks = len(all_docs)
    log(f"Tổng số chunk đọc được từ FAISS index: {total_chunks}")

    char_counts = []
    token_counts = []
    for doc in all_docs:
        text = doc.page_content or ""
        char_counts.append(len(text))
        token_counts.append(count_tokens(text))

    if total_chunks == 0:
        log("Không có chunk nào trong index — kiểm tra lại đường dẫn database_path trong model_config.yml.")
        sys.exit(1)

    avg_chars = statistics.mean(char_counts)
    median_chars = statistics.median(char_counts)
    avg_tokens = statistics.mean(token_counts)
    median_tokens = statistics.median(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    log("")
    log("-" * 70)
    log("KẾT QUẢ THỐNG KÊ")
    log("-" * 70)
    log(f"Số ký tự trung bình / chunk   : {avg_chars:.1f} ký tự  (trung vị: {median_chars:.0f})")
    log(f"Số token trung bình / chunk   : {avg_tokens:.1f} tokens (trung vị: {median_tokens:.0f})")
    log(f"Token nhỏ nhất / lớn nhất     : {min_tokens} / {max_tokens}")
    log("")
    log(">>> Con số cần điền vào Bảng 'Thông tin tổng quát của kho tri thức':")
    log(f">>>   Trung bình số token/chunk: khoảng {avg_tokens:.0f} tokens")
    log("")

    # ── Bước 3: Thời điểm cập nhật gần nhất của FAISS index trên đĩa ──
    try:
        import yaml
        with open("model_config.yml", "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        # db_path thường được định nghĩa trong db_config, không phải model_config;
        # nếu load_vector_db() đã tự biết đường dẫn, ta lấy lại từ đối tượng db nếu có.
    except Exception:
        pass

    # Thử tìm file index.faiss phổ biến trong các vị trí hay gặp
    candidate_paths = []
    for root, dirs, files in os.walk("."):
        for fname in files:
            if fname == "index.faiss":
                candidate_paths.append(os.path.join(root, fname))

    log("-" * 70)
    log("THỜI ĐIỂM CẬP NHẬT FAISS INDEX (dựa trên file index.faiss tìm thấy)")
    log("-" * 70)
    if not candidate_paths:
        log("Không tự động tìm thấy file index.faiss trong thư mục hiện tại và con.")
        log("Hãy tìm thủ công đường dẫn database_path (thường trong model_config.yml")
        log("hoặc db_config trong vectordb.py) rồi kiểm tra ngày sửa đổi file index.faiss")
        log("bằng lệnh: ls -la <đường_dẫn>/index.faiss   (Linux/Mac)")
        log("hoặc:      dir <đường_dẫn>\\index.faiss      (Windows CMD)")
    else:
        for p in candidate_paths:
            mtime = os.path.getmtime(p)
            dt = datetime.datetime.fromtimestamp(mtime)
            log(f"File: {p}")
            log(f">>> Thời điểm cập nhật gần nhất: {dt.strftime('%d/%m/%Y')}")
            log("")

    log("=" * 70)
    log("HOÀN TẤT. Kết quả đã được lưu vào thong_ke_kb_output.txt")
    log("=" * 70)

    with open("thong_ke_kb_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))


if __name__ == "__main__":
    main()