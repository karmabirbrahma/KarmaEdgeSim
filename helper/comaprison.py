import os
import re
import glob
import numpy as np

# --- CONFIGURATION ---
TARGET_DIR = "../sim_results/ite21" # Ensure this points to your log folder
APP = "ALL_APPS_GENERIC"
POLICIES = ['SHO', 'PURE_DDQN']

def main():
    print(f"Scanning directory: {TARGET_DIR}...\n")
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return

    file_pattern = os.path.join(TARGET_DIR, f"SIMRESULT_TWO_TIER_WITH_EO_*_*DEVICES_{APP}.log")
    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print("No log files found.")
        return

    # Dictionary to store all collected values
    data = {'SHO': {'delays': [], 'fails': []}, 
            'PURE_DDQN': {'delays': [], 'fails': []}}

    for file_path in log_files:
        filename = os.path.basename(file_path)
        match = re.search(r"SIMRESULT_TWO_TIER_WITH_EO_(.*)_(\d+)DEVICES_" + APP + r"\.log", filename)
        if not match:
            continue
            
        policy = match.group(1)
        if policy not in POLICIES:
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                stats = lines[1].strip().split(';')
                if len(stats) >= 6:
                    total_tasks = float(stats[0])
                    failed_tasks = float(stats[1])
                    avg_delay = float(stats[4]) # Total Delay / Service Time
                    
                    failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                    
                    data[policy]['delays'].append(avg_delay)
                    data[policy]['fails'].append(failure_rate)

    # 1. Calculate Grand Averages
    sho_avg_delay = np.mean(data['SHO']['delays'])
    sho_avg_fail = np.mean(data['SHO']['fails'])
    
    ddqn_avg_delay = np.mean(data['PURE_DDQN']['delays'])
    ddqn_avg_fail = np.mean(data['PURE_DDQN']['fails'])

    # 2. Calculate Percentage Improvements
    # Formula: ((Baseline - Proposed) / Baseline) * 100
    delay_improvement = ((ddqn_avg_delay - sho_avg_delay) / ddqn_avg_delay) * 100
    fail_improvement = ((ddqn_avg_fail - sho_avg_fail) / ddqn_avg_fail) * 100

    # 3. Print the Final Report
    print("="*50)
    print(" 🏆 OVERALL THESIS METRICS (Averaged 200-2400 Devices)")
    print("="*50)
    
    print("\n--- AVERAGE DELAY (SERVICE TIME) ---")
    print(f"Pure DDQN Overall: {ddqn_avg_delay:.4f} seconds")
    print(f"S-HEO Overall:     {sho_avg_delay:.4f} seconds")
    print(f"🔥 S-HEO is {delay_improvement:.2f}% faster overall.")

    print("\n--- AVERAGE FAILURE RATE ---")
    print(f"Pure DDQN Overall: {ddqn_avg_fail:.4f} %")
    print(f"S-HEO Overall:     {sho_avg_fail:.4f} %")
    print(f"🔥 S-HEO reduces failures by {fail_improvement:.2f}% overall.")
    print("="*50)

if __name__ == "__main__":
    main()