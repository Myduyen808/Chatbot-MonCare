"""
plot_all_results.py
===================
TỰ ĐỘNG vẽ toàn bộ biểu đồ thực nghiệm (Ablation Study & Kiểm định Wilcoxon)
dựa trên số liệu thực tế chính xác từ 4 tập dữ liệu của hệ thống MomCare.
"""
import matplotlib.pyplot as plt
import numpy as np

# thiết lập font chữ tổng thể trực quan rõ ràng
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ==========================================
# DATA ĐƯỢC TRÍCH XUẤT CHÍNH XÁC TỪ LOG CỦA EM
# Cấu trúc mỗi mảng: [Faithfulness, Context Precision, Context Recall, Answer Relevancy]
# ==========================================
methods = ["Dense Only", "BM25 Only", "Hybrid", "Hybrid+Rerank", "Hybrid+Rerank+MQ", "Full System"]

data_kb1 = {
    "faith": [0.800, 0.872, 0.748, 0.856, 0.868, 0.916],
    "prec":  [0.716, 0.796, 0.662, 0.764, 0.790, 0.840],
    "recall":[0.824, 0.896, 0.748, 0.872, 0.876, 0.932],
    "rel":   [0.848, 0.916, 0.790, 0.866, 0.894, 0.954]
}

data_kb2 = {
    "faith": [0.696, 0.624, 0.744, 0.728, 0.784, 0.724],
    "prec":  [0.598, 0.542, 0.662, 0.660, 0.682, 0.658],
    "recall":[0.732, 0.628, 0.790, 0.756, 0.828, 0.760],
    "rel":   [0.690, 0.646, 0.740, 0.768, 0.814, 0.744]
}

data_kb3 = {
    "faith": [0.676, 0.612, 0.608, 0.480, 0.460, 0.652],
    "prec":  [0.550, 0.574, 0.554, 0.414, 0.414, 0.544],
    "recall":[0.740, 0.672, 0.672, 0.532, 0.500, 0.708],
    "rel":   [0.650, 0.644, 0.624, 0.474, 0.458, 0.632]
}

data_vimed = {
    "faith": [0.620, 0.852, 0.692, 0.728, 0.692, 0.708],
    "prec":  [0.604, 0.784, 0.648, 0.672, 0.624, 0.630],
    "recall":[0.680, 0.904, 0.716, 0.776, 0.744, 0.750],
    "rel":   [0.668, 0.856, 0.664, 0.748, 0.684, 0.732]
}

datasets = [
    ("KB1_Medical_Standard", data_kb1),
    ("KB2_Mom_Style (TeenCode)", data_kb2),
    ("KB3_Information_Noise", data_kb3),
    ("ViMedAQA (Chuyên Sâu)", data_vimed)
]

# ==========================================
# ĐỒ THỊ 1: ABLATION STUDY PROGRESSION (LINE CHARTS LAYOUT 2x2)
# ==========================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 11), sharex=False, sharey=True)
axes1 = axes1.flatten()

metrics_meta = [
    ("faith", "Faithfulness", "o", "#1f77b4"),
    ("prec", "Context Precision", "s", "#ff7f0e"),
    ("recall", "Context Recall", "^", "#2ca02c"),
    ("rel", "Answer Relevancy", "d", "#d62728")
]

print("📊 Đang tiến hành vẽ biểu đồ Ablation Study...")
for i, (name, data) in enumerate(datasets):
    ax = axes1[i]
    for key, label, marker, color in metrics_meta:
        ax.plot(methods, data[key], marker=marker, markersize=7, linewidth=2, label=label, color=color)
    
    ax.set_title(f"Tập dữ liệu: {name}", fontsize=13, fontweight='bold', pad=10)
    ax.set_ylim(0.35, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', rotation=15, labelsize=10)
    if i in [0, 2]:
        ax.set_ylabel("Điểm số RAGAS (0.0 - 1.0)", fontsize=11)

# Đặt một Chú thích chung duy nhất cho cả ảnh lớn ở góc trên
handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=4, fontsize=12, shadow=True)
fig1.suptitle("Ablation Study: Tiến Trình Tăng Trưởng Chất Lượng RAG Trên Các Tầng Module MomCare", fontsize=16, fontweight='bold', y=0.99)
fig1.tight_layout(rect=[0, 0.02, 1, 0.93])

# Lưu ảnh đồ thị 1
fig1.savefig("ablation_study_progression.png", dpi=300)
plt.close(fig1)

# ==========================================
# ĐỒ THỊ 2: BASELINE VS FULL SYSTEM COMPARISON (BAR CHARTS LAYOUT 2x2)
# ==========================================
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes2 = axes2.flatten()

db_names = ["KB1", "KB2", "KB3", "ViMedAQA"]
x_indexes = np.arange(len(db_names))
bar_width = 0.35

metrics_bar_meta = [
    ("faith", "Chỉ số: Faithfulness"),
    ("prec", "Chỉ số: Context Precision"),
    ("recall", "Chỉ số: Context Recall"),
    ("rel", "Chỉ số: Answer Relevancy")
]

print("📊 Đang tiến hành vẽ biểu đồ so sánh Kiểm định Thống kê (Wilcoxon)...")
for i, (key, title) in enumerate(metrics_bar_meta):
    ax = axes2[i]
    
    # Gom điểm số của 4 tập ứng với phương pháp 1 và phương pháp 6
    baseline_scores = [data_kb1[key][0], data_kb2[key][0], data_kb3[key][0], data_vimed[key][0]]
    full_sys_scores = [data_kb1[key][5], data_kb2[key][5], data_kb3[key][5], data_vimed[key][5]]
    
    # Vẽ các cột nhóm
    rects1 = ax.bar(x_indexes - bar_width/2, baseline_scores, bar_width, label='Baseline (Dense Only)', color='#aec7e8', edgecolor='gray', alpha=0.9)
    rects2 = ax.bar(x_indexes + bar_width/2, full_sys_scores, bar_width, label='Full System (MomCare)', color='#1f77b4', edgecolor='gray', alpha=0.9)
    
    # Thêm số điểm lên đầu cột để dễ quan sát số liệu
    ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(db_names, fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    if i in [0, 2]:
        ax.set_ylabel("Điểm số trung bình", fontsize=11)

handles2, labels2 = axes2[0].get_legend_handles_labels()
fig2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=12, shadow=True)
fig2.suptitle("So Sánh Hiệu Năng Giữa Hệ Thống Toàn Diện (Full System) và Bản Cơ Sở (Baseline) Trên 4 Tập Dữ Liệu", fontsize=16, fontweight='bold', y=0.99)
fig2.tight_layout(rect=[0, 0.02, 1, 0.93])

# Lưu ảnh đồ thị 2
fig2.savefig("baseline_vs_full_system.png", dpi=300)
plt.close(fig2)

print("🎉 XONG! Tất cả các biểu đồ chất lượng cao đã được lưu thành công tại thư mục dự án:")
print(" ➡️ ablation_study_progression.png")
print(" ➡️ baseline_vs_full_system.png")