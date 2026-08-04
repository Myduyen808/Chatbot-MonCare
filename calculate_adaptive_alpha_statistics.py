from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


INPUT_FILE = Path("adaptive_alpha_v2_detailed.csv")
OUTPUT_FILE = Path("adaptive_alpha_v2_statistics.csv")

PAIR_KEYS = [
    "dataset",
    "profile",
    "question",
]


def run_wilcoxon() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {INPUT_FILE}"
        )

    detailed = pd.read_csv(INPUT_FILE)

    required_columns = {
        "config",
        "dataset",
        "profile",
        "question",
        "hit_at_5",
        "rr_at_5",
    }

    missing_columns = (
        required_columns - set(detailed.columns)
    )

    if missing_columns:
        raise ValueError(
            "File kết quả thiếu các cột: "
            + ", ".join(sorted(missing_columns))
        )

    adaptive_rows = (
        detailed[
            detailed["config"]
            == "Adaptive alpha v2"
        ][
            PAIR_KEYS + ["hit_at_5", "rr_at_5"]
        ]
        .rename(
            columns={
                "hit_at_5": "adaptive_hit",
                "rr_at_5": "adaptive_rr",
            }
        )
    )

    statistics_rows = []

    for fixed_config in [
        "Fixed alpha = 0.3",
        "Fixed alpha = 0.4",
        "Fixed alpha = 0.5",
        "Fixed alpha = 0.7",
    ]:
        fixed_rows = (
            detailed[
                detailed["config"] == fixed_config
            ][
                PAIR_KEYS
                + ["hit_at_5", "rr_at_5"]
            ]
            .rename(
                columns={
                    "hit_at_5": "fixed_hit",
                    "rr_at_5": "fixed_rr",
                }
            )
        )

        paired = adaptive_rows.merge(
            fixed_rows,
            on=PAIR_KEYS,
            how="inner",
            validate="one_to_one",
        )

        if paired.empty:
            print(
                f"⚠️ Bỏ qua {fixed_config}: "
                "không có dữ liệu ghép cặp."
            )
            continue

        for metric_name, adaptive_column, fixed_column in [
            (
                "MRR@5",
                "adaptive_rr",
                "fixed_rr",
            ),
            (
                "Hit Rate@5",
                "adaptive_hit",
                "fixed_hit",
            ),
        ]:
            differences = (
                paired[adaptive_column]
                - paired[fixed_column]
            )

            if (differences != 0).any():
                test_result = wilcoxon(
                    paired[adaptive_column],
                    paired[fixed_column],
                    zero_method="pratt",
                    alternative="two-sided",
                )

                statistic = float(
                    test_result.statistic
                )

                p_value = float(
                    test_result.pvalue
                )
            else:
                statistic = 0.0
                p_value = 1.0

            statistics_rows.append(
                {
                    "comparison": (
                        "Adaptive alpha v2 vs "
                        + fixed_config
                    ),
                    "metric": metric_name,
                    "n_pairs": len(paired),
                    "adaptive_mean": (
                        paired[adaptive_column].mean()
                    ),
                    "fixed_mean": (
                        paired[fixed_column].mean()
                    ),
                    "mean_difference": (
                        differences.mean()
                    ),
                    "better_cases": int(
                        (differences > 0).sum()
                    ),
                    "equal_cases": int(
                        (differences == 0).sum()
                    ),
                    "worse_cases": int(
                        (differences < 0).sum()
                    ),
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant_0_05": (
                        p_value < 0.05
                    ),
                }
            )

    statistics_df = pd.DataFrame(
        statistics_rows
    )

    statistics_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n📊 Kết quả kiểm định Wilcoxon:")
    print(statistics_df.to_string(index=False))
    print(f"\n✅ Đã lưu: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_wilcoxon()