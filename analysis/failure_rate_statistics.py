import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("../sim_results")

START_ITER = 31
END_ITER = 45

NUM_DEVICES = "2400"
APP_TYPE = "ALL_APPS_GENERIC"

POLICIES = {
    "SHO": "HEO",
    "PURE_DDQN": "DDQN",
    "DDQN_MOB_ONLY": "DDQN_Mobility",
    "FUZZY_BASED": "Fuzzy_Based",
}

OUTPUT_DIR = Path("failure_statistics")

OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV = OUTPUT_DIR / "failure_rate_raw.csv"

# ============================================================
# Helper Functions
# ============================================================

def build_filename(policy):
    return (
        f"SIMRESULT_TWO_TIER_WITH_EO_"
        f"{policy}_{NUM_DEVICES}DEVICES_{APP_TYPE}.log"
    )


def extract_failure_rate(log_file):
    """
    Returns failure rate (%) from a simulator log.

    stats[0] = total tasks
    stats[1] = failed tasks
    """

    with open(log_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise RuntimeError(f"Invalid log file: {log_file}")

    stats = lines[1].strip().split(";")

    completed = float(stats[0])
    failed = float(stats[1])
    total_tasks = completed + failed

    if total_tasks == 0:
        return np.nan

    return (failed / total_tasks) * 100.0


# ============================================================
# Scan Iterations
# ============================================================

rows = []

print("=" * 70)
print("Extracting Failure Rates")
print("=" * 70)

for iteration in range(START_ITER, END_ITER + 1):

    ite_folder = BASE_DIR / f"ite{iteration}"

    if not ite_folder.exists():
        print(f"Skipping missing folder: {ite_folder}")
        continue

    row = {"Iteration": iteration}

    for policy, label in POLICIES.items():

        logfile = ite_folder / build_filename(policy)

        if not logfile.exists():
            print(f"Missing: {logfile}")
            row[label] = np.nan
            continue

        try:
            row[label] = extract_failure_rate(logfile)

        except Exception as e:
            print(e)
            row[label] = np.nan

    rows.append(row)

df = pd.DataFrame(rows)

print("\nRaw Failure Rates\n")
print(df)

df.to_csv(RAW_CSV, index=False)

print("\nSaved:", RAW_CSV)