import csv
import os

file_path = r"data_store\csv\qa_me_bim_sau_sinh_100.csv"

if not os.path.exists(file_path):
    print(f"Không tìm thấy file tại: {file_path}")
else:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader, 1):
                if not row:
                    print(f"Lỗi tại dòng {i}: Dòng trống")
        print("Đã kiểm tra xong, không thấy lỗi định dạng nghiêm trọng.")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")