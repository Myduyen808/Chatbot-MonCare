"""
run_significance_test.py
=====================
TỰ ĐỘNG đọc trực tiếp điểm số chi tiết từng câu từ các file checkpoint JSON
(scores_final_*.json) và chạy Wilcoxon Signed-Rank Test chuẩn xác cho cả 4 tập dữ liệu,
sau đó xuất thẳng ra mã bảng LaTeX.
Chạy: python run_significance_test.py
"""
import os
import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ALPHA = 0.05
DATASETS = ["KB1_Standard", "KB2_TeenCode", "KB3_Noise", "ViMedAQA"]
METRICS = {
    "faith": "Faithfulness",
    "precision": "Context Precision",
    "recall": "Context Recall",
    "relevancy": "Answer Relevancy"
}

def run_wilcoxon(base_scores, full_scores):
    """Chạy kiểm định Wilcoxon Signed-Rank Test trên mảng điểm chi tiết của 25 câu hỏi"""
    base_scores = np.array(base_scores)
    full_scores = np.array(full_scores)
    
    mean_base = np.mean(base_scores)
    mean_full = np.mean(full_scores)
    delta = mean_full - mean_base
    
    # Kiểm tra xem có sự khác biệt nào giữa 2 mảng không để tránh lỗi kiểm định
    if np.all(full_scores == base_scores) or np.count_nonzero(full_scores - base_scores) == 0:
        p_value = 1.0
    else:
        # Kiểm định một đuôi (alternative='greater') để chứng minh Full System tốt hơn Baseline
        try:
            _, p_value = stats.wilcoxon(full_scores, base_scores, alternative="greater")
        except Exception:
            p_value = 1.0
            
    sig = "$p < 0.05$ $\\checkmark$" if p_value < ALPHA else "Không đạt"
    return mean_base, mean_full, delta, p_value, sig

def main():
    latex_lines = []
    latex_lines.append("\\begin{table}[H]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Kết quả kiểm định ý nghĩa thống kê Wilcoxon Signed-Rank Test giữa Full System và Baseline}")
    latex_lines.append("\\label{tab:significance_test}\n\\renewcommand{\\arraystretch}{1.4}")
    latex_lines.append("\\begin{tabular}{|l|l|c|c|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Tập dữ liệu} & \\textbf{Chỉ số} & \\textbf{Baseline} & \\textbf{Full Sys} & \\textbf{$\\Delta$} & \\textbf{p-value} & \\textbf{Kết luận} \\\\ \\hline")

    all_significant = True
    any_file_found = False

    for dataset in DATASETS:
        score_file = f"scores_final_{dataset}.json"
        if not os.path.exists(score_file):
            print(f"⏭️ Bỏ qua tập {dataset} vì chưa có file kết quả điểm {score_file}")
            continue
            
        any_file_found = True
        with open(score_file, "r", encoding="utf-8") as f:
            score_data = json.load(f)
            
        base_data = score_data.get("1. Dense Only", {})
        full_data = score_data.get("6. Full System", {})
        
        if not base_data or not full_data:
            print(f"⚠️ Tập {dataset} thiếu dữ liệu của Baseline hoặc Full System.")
            continue
            
        first_row = True
        for metric_key, metric_name in METRICS.items():
            b_scores = base_data.get(metric_key, [])
            f_scores = full_data.get(metric_key, [])
            
            if len(b_scores) == 0 or len(f_scores) == 0:
                print(f"⚠️ Chỉ số {metric_name} ở tập {dataset} bị trống điểm.")
                continue
                
            m_base, m_full, delta, p_val, sig = run_wilcoxon(b_scores, f_scores)
            
            if "Không đạt" in sig:
                all_significant = False
                
            kb_col = dataset if first_row else ""
            latex_lines.append(f"{kb_col} & {metric_name} & {m_base:.3f} & {m_full:.3f} & {delta:+.3f} & {p_val:.4f} & {sig} \\\\")
            first_row = False
        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    if not any_file_found:
        print("❌ LỖI: Không tìm thấy bất kỳ file 'scores_final_*.json' nào trong cùng thư mục!")
        print("Vui lòng đảm bảo em đã chạy xong file 'evaluate_results.py' trước để sinh điểm.")
        return

    # In kết quả mã bảng ra màn hình Terminal
    print("\n" + "="*80)
    print("✅ KẾT QUẢ KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (WILCOXON SIGNED-RANK TEST) CHUẨN")
    print("="*80 + "\n")
    print("\n".join(latex_lines))

    # Lưu mã LaTeX ra file xịn
    with open("significance_test_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print("\n" + "="*80)
    if all_significant:
        print("🎉 TUYỆT VỜI: Toàn bộ các chỉ số so sánh đều đạt ý nghĩa thống kê p < 0.05!")
    else:
        print("⚠️ LƯU Ý: Có một số chỉ số cải thiện chưa đủ đạt mốc ý nghĩa thống kê p < 0.05.")
    print("Đã lưu bảng mã LaTeX chuẩn tại file: significance_test_table.tex")
    print("="*80)

if __name__ == "__main__":
    main()