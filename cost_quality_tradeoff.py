"""
cost_quality_tradeoff.py
=========================
Trả lời đúng câu hỏi của thầy: "1$ đổi được bao nhiêu Faithfulness?"

Đọc dữ liệu THẬT đã đo được từ real_benchmark_4_configs.py (không có số liệu
hardcode/mô phỏng nào), quy đổi Token -> USD theo ĐÚNG bảng giá Groq hiện hành
cho llama-3.1-8b-instant, tính chỉ số hiệu suất sinh lời FpD (Faithfulness per
Dollar), và vẽ Pareto Frontier (Cost vs Faithfulness) cho 4 cấu hình.

⚠️ GIÁ GROQ DÙNG TRONG SCRIPT NÀY (kiểm tra lại tại https://groq.com/pricing
   trước khi nộp báo cáo, vì giá API có thể thay đổi theo thời gian):
     llama-3.1-8b-instant:
       - Input  (prompt tokens):     $0.05 / 1,000,000 tokens
       - Output (completion tokens): $0.08 / 1,000,000 tokens
   (Input và Output có giá KHÁC NHAU — đây là lý do real_benchmark_4_configs.py
   đã được sửa để lưu riêng prompt_tokens và completion_tokens thay vì gộp
   chung "tokens", để quy đổi USD ở đây được chính xác.)

Cách chạy:
    python cost_quality_tradeoff.py

Input: tự động dò các file theo thứ tự ưu tiên:
    1. telemetry_all_seeds.csv       (ưu tiên nhất — dữ liệu nhiều seed, đáng tin nhất)
    2. telemetry_raw_log_seed*.csv   (gộp tất cả các file seed tìm thấy)
    3. telemetry_raw_log.csv         (bản chạy đơn cũ, không seed)

Output:
    - In bảng: Cost trung bình (USD), Faithfulness trung bình, FpD = Faithfulness/Cost
    - Lưu ảnh 'Pareto_Frontier_Chart.png'
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # để chạy được kể cả không có màn hình (server/headless)
import matplotlib.pyplot as plt

# ==============================================================================
# GIÁ GROQ CHO llama-3.1-8b-instant (USD / 1 TRIỆU TOKEN)
# Nguồn: https://groq.com/pricing — kiểm tra lại trước khi nộp báo cáo vì giá
# API có thể đổi theo thời gian, khác thời điểm bạn đọc script này.
# ==============================================================================
PRICE_PER_1M_INPUT_TOKENS = 0.05   # USD / 1,000,000 prompt tokens
PRICE_PER_1M_OUTPUT_TOKENS = 0.08  # USD / 1,000,000 completion tokens

CONFIG_NAMES = {
    "A": "Vanilla RAG (thô)",
    "B": "Nén cực hạn",
    "C": "Tối ưu một nửa",
    "D": "MomCare Full",
}
CONFIG_ORDER = ["A", "B", "C", "D"]
CONFIG_COLORS = {"A": "#e74c3c", "B": "#f39c12", "C": "#3498db", "D": "#2ecc71"}


def load_telemetry():
    """
    Tự động dò file dữ liệu THẬT theo thứ tự ưu tiên. KHÔNG tự tạo dữ liệu giả
    nếu thiếu file — dừng luôn và báo rõ để bạn biết cần chạy benchmark trước.
    """
    if os.path.exists("telemetry_all_seeds.csv"):
        df = pd.read_csv("telemetry_all_seeds.csv")
        print(f"✅ Đã nạp 'telemetry_all_seeds.csv' ({len(df)} dòng, nhiều seed).")
        return df

    seed_files = sorted(glob.glob("telemetry_raw_log_seed*.csv"))
    if seed_files:
        dfs = []
        for f in seed_files:
            d = pd.read_csv(f)
            d["source_file"] = f
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Đã gộp {len(seed_files)} file seed: {seed_files} -> {len(df)} dòng.")
        return df

    if os.path.exists("telemetry_raw_log.csv"):
        df = pd.read_csv("telemetry_raw_log.csv")
        print(f"✅ Đã nạp 'telemetry_raw_log.csv' ({len(df)} dòng, 1 lần chạy đơn).")
        print("⚠️ Đây chỉ là 1 lần chạy — độ tin cậy thấp hơn multi-seed. "
              "Khuyến nghị chạy run_multi_seed_experiment() để có kết quả ổn định hơn.")
        return df

    raise FileNotFoundError(
        "❌ Không tìm thấy file telemetry nào (telemetry_all_seeds.csv / "
        "telemetry_raw_log_seed*.csv / telemetry_raw_log.csv) trong thư mục hiện tại.\n"
        "   Hãy chạy real_benchmark_4_configs.py trước để tạo dữ liệu thật."
    )


def compute_cost_usd(row):
    """
    Quy đổi 1 lượt đo (1 câu hỏi, 1 cấu hình) ra USD theo giá Groq thật.
    Nếu file cũ không có cột prompt_tokens/completion_tokens tách riêng
    (bản chạy trước khi sửa), coi toàn bộ "tokens" là input (ước lượng
    CẬN TRÊN của chi phí thật, vì input thường rẻ hơn output).
    """
    if "prompt_tokens" in row and "completion_tokens" in row and not pd.isna(row.get("prompt_tokens")):
        input_tok = row["prompt_tokens"]
        output_tok = row["completion_tokens"]
    else:
        input_tok = row["tokens"]
        output_tok = 0
    cost = (input_tok / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS + \
           (output_tok / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
    return cost


def is_pareto_efficient(costs, scores):
    """
    Trả về mảng boolean: True nếu điểm đó nằm TRÊN biên Pareto (không bị điểm
    nào khác "thống trị" — tức không có điểm nào vừa RẺ HƠN vừa CHẤT LƯỢNG
    CAO HƠN cùng lúc).
    """
    n = len(costs)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j thống trị i nếu j rẻ hơn hoặc bằng VÀ chất lượng cao hơn hoặc bằng,
            # và ít nhất 1 tiêu chí thực sự tốt hơn (không phải trùng hệt nhau)
            if costs[j] <= costs[i] and scores[j] >= scores[i] and \
               (costs[j] < costs[i] or scores[j] > scores[i]):
                efficient[i] = False
                break
    return efficient


def main():
    df = load_telemetry()

    if "config" not in df.columns:
        print("❌ File dữ liệu thiếu cột 'config'. Kiểm tra lại file telemetry đầu vào.")
        sys.exit(1)

    df["cost_usd"] = df.apply(compute_cost_usd, axis=1)

    # ==========================================================================
    # BẢNG TỔNG HỢP: Cost trung bình, Faithfulness trung bình, FpD = F / Cost
    # ==========================================================================
    summary_rows = []
    for cfg in CONFIG_ORDER:
        sub = df[df["config"] == cfg]
        if sub.empty:
            continue
        n = len(sub)
        avg_cost = sub["cost_usd"].mean()
        std_cost = sub["cost_usd"].std()
        avg_faith = sub["faithfulness"].mean()
        std_faith = sub["faithfulness"].std()
        avg_latency = sub["latency_ms"].mean()
        # FpD: bao nhiêu điểm Faithfulness đổi được trên mỗi 1 USD chi ra.
        # Tránh chia cho 0 nếu cost quá nhỏ (không nên xảy ra với dữ liệu thật).
        fpd = avg_faith / avg_cost if avg_cost > 0 else float("inf")
        summary_rows.append(dict(
            config=cfg, name=CONFIG_NAMES[cfg], n=n,
            avg_cost_usd=avg_cost, std_cost_usd=std_cost,
            avg_faithfulness=avg_faith, std_faithfulness=std_faith,
            avg_latency_ms=avg_latency,
            faithfulness_per_dollar=fpd,
            cost_per_1000_queries_usd=avg_cost * 1000,
        ))

    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 100)
    print(" PHÂN TÍCH ĐÁNH ĐỔI CHI PHÍ - CHẤT LƯỢNG (COST-QUALITY TRADEOFF)")
    print(f" Giá dùng: input ${PRICE_PER_1M_INPUT_TOKENS}/1M tokens, output ${PRICE_PER_1M_OUTPUT_TOKENS}/1M tokens (Groq, llama-3.1-8b-instant)")
    print("=" * 100)
    print(f"{'Cấu hình':<22} | {'n':<4} | {'Cost/lượt (USD)':<18} | {'Faithfulness':<16} | "
          f"{'FpD (F/$)':<12} | {'Cost/1000 lượt':<15}")
    print("-" * 100)
    for _, r in summary.iterrows():
        print(f"{r['name']:<22} | {r['n']:<4} | "
              f"${r['avg_cost_usd']:.6f}±{r['std_cost_usd']:.6f} | "
              f"{r['avg_faithfulness']:.3f}±{r['std_faithfulness']:.3f}    | "
              f"{r['faithfulness_per_dollar']:.1f}      | "
              f"${r['cost_per_1000_queries_usd']:.2f}")
    print("-" * 100)

    # Trả lời trực tiếp câu hỏi của thầy: "1$ đổi được bao nhiêu Faithfulness?"
    print("\n📌 TRẢ LỜI CÂU HỎI CỦA THẦY — '1$ đổi được bao nhiêu Faithfulness?':")
    for _, r in summary.iterrows():
        queries_per_dollar = 1.0 / r["avg_cost_usd"] if r["avg_cost_usd"] > 0 else float("inf")
        print(f"   • {r['name']}: 1 USD chạy được ~{queries_per_dollar:,.0f} lượt hỏi, "
              f"mỗi lượt trung bình đạt {r['avg_faithfulness']:.3f} điểm faithfulness "
              f"-> tổng {r['faithfulness_per_dollar']:.1f} điểm faithfulness/1 USD.")

    # ==========================================================================
    # XÁC ĐỊNH BIÊN PARETO
    # ==========================================================================
    costs = summary["avg_cost_usd"].values
    scores = summary["avg_faithfulness"].values
    pareto_mask = is_pareto_efficient(costs, scores)
    summary["on_pareto_frontier"] = pareto_mask

    print("\n📊 CẤU HÌNH NẰM TRÊN BIÊN PARETO (không bị cấu hình nào khác vừa rẻ hơn")
    print("   vừa chất lượng cao hơn cùng lúc):")
    for _, r in summary[summary["on_pareto_frontier"]].iterrows():
        print(f"   ✅ {r['name']} (cost=${r['avg_cost_usd']:.6f}, faithfulness={r['avg_faithfulness']:.3f})")
    dominated = summary[~summary["on_pareto_frontier"]]
    if not dominated.empty:
        print("\n⚠️ CẤU HÌNH BỊ THỐNG TRỊ (có lựa chọn khác tốt hơn ở CẢ 2 tiêu chí):")
        for _, r in dominated.iterrows():
            print(f"   ❌ {r['name']} (cost=${r['avg_cost_usd']:.6f}, faithfulness={r['avg_faithfulness']:.3f})")

    # ==========================================================================
    # VẼ PARETO FRONTIER
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for _, r in summary.iterrows():
        cfg = r["config"]
        marker = "o" if r["on_pareto_frontier"] else "x"
        size = 260 if r["on_pareto_frontier"] else 180
        ax.scatter(r["avg_cost_usd"], r["avg_faithfulness"],
                   s=size, marker=marker, color=CONFIG_COLORS[cfg],
                   edgecolors="black", linewidths=1.2, zorder=3,
                   label=f"{cfg}: {r['name']}")
        ax.annotate(
            f"{cfg}\n(${r['avg_cost_usd']:.5f}, {r['avg_faithfulness']:.2f})",
            (r["avg_cost_usd"], r["avg_faithfulness"]),
            textcoords="offset points", xytext=(10, 8), fontsize=9,
        )

    # Nối các điểm trên biên Pareto bằng đường nét đứt, theo thứ tự cost tăng dần
    frontier = summary[summary["on_pareto_frontier"]].sort_values("avg_cost_usd")
    if len(frontier) > 1:
        ax.plot(frontier["avg_cost_usd"], frontier["avg_faithfulness"],
                linestyle="--", color="gray", linewidth=1.5, zorder=2,
                label="Pareto Frontier")

    ax.set_xlabel("Chi phí trung bình mỗi lượt hỏi (USD)", fontsize=11)
    ax.set_ylabel("Faithfulness trung bình (LLM-judge)", fontsize=11)
    ax.set_title("Pareto Frontier: Đánh đổi Chi phí - Chất lượng giữa 4 cấu hình RAG",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig("Pareto_Frontier_Chart.png", dpi=150)
    print("\n✅ Đã lưu biểu đồ 'Pareto_Frontier_Chart.png'.")

    summary.to_csv("cost_quality_summary.csv", index=False, encoding="utf-8-sig")
    print("✅ Đã lưu bảng tổng hợp vào 'cost_quality_summary.csv'.")
    print("=" * 100)


if __name__ == "__main__":
    main()