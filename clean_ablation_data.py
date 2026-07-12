import re

DATASETS = ["log_KB1_Standard.txt", "log_KB2_TeenCode.txt", "log_KB3_Noise.txt", "log_ViMedAQA.txt"]

def parse_log(file_path):
    methods = {}
    dataset_name = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "CHẠY THỰC NGHIỆM TRÊN TẬP:" in line:
                dataset_name = line.split("TẬP:")[1].split("(")[0].strip()
                methods[dataset_name] = {}
            elif "|" in line and not line.startswith("---") and not line.startswith("Method"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 6:
                    try:
                        method = parts[0]
                        f = float(parts[1]); p = float(parts[2]); r = float(parts[3]); a = float(parts[4]); l = float(parts[5])
                        
                        # NẾU BỎ CÁC HÀNG LỖI 0.000 ĐỂ LỌC BỎ RA KHỎI BẢNG
                        if f == 0.0 and p == 0.0 and r == 0.0 and a == 0.0:
                            continue # Bỏ qua hàng lỗi 100%
                            
                        if method not in methods[dataset_name]:
                            methods[dataset_name][method] = {"faith": [], "precision": [], "recall": [], "relevancy": [], "latency": []}
                        
                        methods[dataset_name][method]["faith"].append(f)
                        methods[temperature]["precision"].append(p)
                        methods[method]["recall"].append(r)
                        methods[method]["relevancy"].append(a)
                        methods[method]["latency"].append(l)
                    except ValueError:
                        pass
    return dataset_name, methods

print(f"{'Method':<25} | {'Dataset':<15} | {'Faithfulness':<12} | {'Context Prec':<12} | {'Context Recall':<12} | {'Answer Rel.':<12} | {'Latency (s)':<10} | {'N (hợp lệ)'}")
print("-"*105)

for ds_name, ds_data in DATASETS.items():
    if not ds_data: continue
    n_total = len(ds_data["1. Dense Only"]["faith"]) # Dùng Dense only để tính tổng số câu
    
    for method_name, m_data in ds_data.items():
        n_valid = len(m_data["faith"])
        avg_f = sum(m_data["faith"]) / n_valid if n_valid > 0 else 0.0
        avg_p = sum(m_data["precision"]) / n_valid if n_valid > 0 else 0.0
        avg_r = sum(m_data["recall"]) / n_valid if n_valid > 0 else 0.0
        avg_a = sum(m_data["relevancy"]) / n_valid if n_valid > 0 else 0.0
        avg_l = sum(m_data["latency"]) / n_valid if n_valid > 0 else 0.0
        
        print(f"{method_name:<25} | {ds_name:<15} | {avg_f:<12.3f} | {avg_p:<12.3f} | {avg_r:<12.3f} | {avg_a:<12.3f} | {avg_l:<10.2f} | {n_valid}/{n_total} ({n_valid/n_total*100:.0f}%)")