from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("evaluation/vimedaqa_clean/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_vi(value):
    return f"{value:.2f}".replace(".", ",")


# =========================================================
# HÌNH 1: SO SÁNH AVERAGE
# =========================================================

models = [
    "Llama2-7B",
    "ViGPT",
    "PhoGPT-4B",
    "Gemma-2B",
    "VinaLlama-2.7B",
    "Llama3",
    "Gemma-7B",
    "MomCare RAG",
    "VinaLlama-7B",
]

averages = [
    24.32,
    37.17,
    50.13,
    50.84,
    52.72,
    55.05,
    55.05,
    55.21,
    56.89,
]

colors = [
    "#A9B0BA" if model != "MomCare RAG" else "#C74440"
    for model in models
]

fig, ax = plt.subplots(figsize=(9.2, 5.6))

bars = ax.barh(
    models,
    averages,
    color=colors,
    edgecolor="white",
    height=0.68,
)

ax.set_xlabel("Điểm Average")
ax.set_xlim(0, 65)
ax.grid(axis="x", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

for bar, value in zip(bars, averages):
    ax.text(
        value + 0.5,
        bar.get_y() + bar.get_height() / 2,
        format_vi(value),
        va="center",
        fontsize=9,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "vimedaqa_average_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =========================================================
# HÌNH 2: RETRIEVAL HIT@5 VÀ MISS@5
# =========================================================

metrics = [
    "BERTScore",
    "BLEU",
    "METEOR",
    "ROUGE-L",
    "Average",
]

hit_values = [
    84.57,
    36.18,
    63.37,
    56.47,
    60.15,
]

miss_values = [
    70.39,
    7.17,
    20.31,
    18.77,
    29.16,
]

x = np.arange(len(metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(9.2, 5.4))

hit_bars = ax.bar(
    x - width / 2,
    hit_values,
    width,
    label="Retrieval Hit@5",
    color="#3B78B5",
)

miss_bars = ax.bar(
    x + width / 2,
    miss_values,
    width,
    label="Retrieval Miss@5",
    color="#D08363",
)

ax.set_ylabel("Điểm")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

for bars in (hit_bars, miss_bars):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            format_vi(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "vimedaqa_hit_miss_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Đã lưu biểu đồ tại: {OUTPUT_DIR.resolve()}")