from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    sem,
    t,
    ttest_rel,
    f_oneway,
)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("failure_statistics")
OUTPUT_DIR = INPUT_DIR

RAW_FILE = INPUT_DIR / "failure_rate_raw.csv"

SUMMARY_FILE = OUTPUT_DIR / "summary_statistics.csv"
TUKEY_FILE = OUTPUT_DIR / "tukey_results.csv"
ANOVA_FILE = OUTPUT_DIR / "anova_results.txt"
PAIR_FILE = OUTPUT_DIR / "heo_vs_ddqn_statistics.csv"

ALGORITHMS = [
    "HEO",
    "DDQN",
    "DDQN_Mobility",
    "Fuzzy_Based",
]

ALPHA = 0.05


# ============================================================
# Load Data
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Cannot find {RAW_FILE}")

df = pd.read_csv(RAW_FILE)

print("=" * 70)
print("Failure Rate Statistical Analysis")
print("=" * 70)
print(df)
print()


# ============================================================
# 1. Summary Statistics + 95% Confidence Interval
# ============================================================

summary = []

for algo in ALGORITHMS:

    values = df[algo].dropna()

    n = len(values)

    mean = values.mean()

    std = values.std(ddof=1)

    stderr = sem(values)

    t_critical = t.ppf(0.975, n - 1)

    margin = t_critical * stderr

    ci_lower = mean - margin
    ci_upper = mean + margin

    summary.append({
        "Algorithm": algo,
        "N": n,
        "Mean": mean,
        "Std": std,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "Margin of Error": margin,
    })

summary_df = pd.DataFrame(summary)

summary_df = summary_df.round(6)

summary_df.to_csv(SUMMARY_FILE, index=False)

print("=" * 70)
print("Summary Statistics")
print("=" * 70)
print(summary_df)
print()


# ============================================================
# 2. One-Way ANOVA
# ============================================================

groups = [df[a].dropna() for a in ALGORITHMS]

F, p = f_oneway(*groups)

with open(ANOVA_FILE, "w") as f:

    f.write("One-Way ANOVA\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"F-statistic : {F:.6f}\n")
    f.write(f"p-value     : {p:.6f}\n\n")

    if p < ALPHA:
        f.write(
            "Conclusion:\n"
            "There is a statistically significant difference "
            "among the algorithms (p < 0.05).\n"
        )
    else:
        f.write(
            "Conclusion:\n"
            "No statistically significant difference "
            "among the algorithms (p >= 0.05).\n"
        )

print("=" * 70)
print("One-Way ANOVA")
print("=" * 70)
print(f"F = {F:.6f}")
print(f"p = {p:.6f}")
print()


# ============================================================
# 3. Tukey HSD
# ============================================================

long_df = pd.melt(
    df,
    id_vars=["Iteration"],
    value_vars=ALGORITHMS,
    var_name="Algorithm",
    value_name="FailureRate",
)

tukey = pairwise_tukeyhsd(
    endog=long_df["FailureRate"],
    groups=long_df["Algorithm"],
    alpha=0.05,
)

tukey_table = pd.DataFrame(
    tukey.summary().data[1:],
    columns=tukey.summary().data[0]
)

tukey_table.to_csv(TUKEY_FILE, index=False)

print("=" * 70)
print("Tukey HSD")
print("=" * 70)
print(tukey)
print()


# ============================================================
# 4. Paired t-test (HEO vs DDQN)
# ============================================================

heo = df["HEO"]
ddqn = df["DDQN"]

t_stat, p_value = ttest_rel(heo, ddqn)

difference = heo - ddqn

mean_difference = difference.mean()

std_difference = difference.std(ddof=1)

stderr = sem(difference)

t_critical = t.ppf(0.975, len(difference) - 1)

margin = t_critical * stderr

ci_lower = mean_difference - margin
ci_upper = mean_difference + margin

pair_df = pd.DataFrame({
    "Statistic": [
        "Mean Difference",
        "t-statistic",
        "p-value",
        "95% CI Lower",
        "95% CI Upper",
    ],
    "Value": [
        mean_difference,
        t_stat,
        p_value,
        ci_lower,
        ci_upper,
    ]
})

pair_df = pair_df.round(6)

pair_df.to_csv(PAIR_FILE, index=False)

print("=" * 70)
print("Paired t-test (HEO vs DDQN)")
print("=" * 70)
print(pair_df)
print()

if p_value < ALPHA:
    print("Result:")
    print("HEO significantly improves the failure rate over DDQN.")
else:
    print("Result:")
    print("No statistically significant difference between HEO and DDQN.")

print()
print("=" * 70)
print("Analysis Complete")
print("=" * 70)
print(f"Summary Statistics : {SUMMARY_FILE}")
print(f"ANOVA              : {ANOVA_FILE}")
print(f"Tukey HSD          : {TUKEY_FILE}")
print(f"HEO vs DDQN        : {PAIR_FILE}")