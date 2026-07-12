# plot_variance.py
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--kb", required=True)
args = parser.parse_args()

folder = f"variance_runs_{args.kb}"
files = sorted(glob.glob(f"{folder}/{args.kb}_run*.csv"))

if not files:
    print(f"Không tìm thấy file nào trong {folder}/")
    exit()

all_runs = []
for f in files:
    df = pd.read_csv(f)
    df["run"] = os.path.basename(f)
    all_runs.append(df)

combined = pd.concat(all_runs, ignore_index=True)

metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
metrics = [m for m in metrics if m in combined.columns]

# 1. Boxplot
fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
if len(metrics) == 1:
    axes = [axes]

for ax, metric in zip(axes, metrics):
    data = [combined[combined["run"] == os.path.basename(f)][metric].dropna().tolist() for f in files]
    ax.boxplot(data, labels=[f"Run {i+1}" for i in range(len(files))])
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.axhline(y=combined[metric].mean(), color='r', linestyle='--', alpha=0.5, label='Mean')
    ax.legend()

plt.suptitle(f"RAGAS Variance Analysis — {args.kb.upper()}", fontsize=13, fontweight='bold')
plt.tight_layout()
output_path = f"{folder}/boxplot_{args.kb}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Đã lưu boxplot: {output_path}")
plt.show()

# 2. Bảng tổng hợp mean ± std
print("\n📊 Kết quả tổng hợp 5 lần chạy:")
print(f"{'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 55)
for metric in metrics:
    vals = combined[metric].dropna()
    print(f"{metric:<25} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f}")

# 3. Lưu CSV tổng hợp
summary_rows = []
for metric in metrics:
    vals = combined[metric].dropna()
    summary_rows.append({
        "metric": metric,
        "mean": round(vals.mean(), 4),
        "std": round(vals.std(), 4),
        "min": round(vals.min(), 4),
        "max": round(vals.max(), 4),
        "cv_percent": round(vals.std() / vals.mean() * 100, 2) if vals.mean() > 0 else 0
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = f"{folder}/summary_{args.kb}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\n✅ Đã lưu bảng tổng hợp: {summary_path}")
print("\n📋 Chi tiết:")
print(summary_df.to_string(index=False))