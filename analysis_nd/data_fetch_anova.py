import os
import glob
import re
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- CONFIGURATION ---
BASE_DIR = "../sim_results" 
ITERATION_START = 31
ITERATION_END = 45  

TARGET_APP = "ALL_APPS_GENERIC"

POLICIES = {
    'SHO': 'HEO', 
    'PURE_DDQN': 'DDQN', 
    'DDQN_MOB_ONLY': 'DDQN_Mobility', 
    'FUZZY_BASED': 'Fuzzy_Based'
}

COMPLETED_TASKS_COL = 0      # 1st value 
FAILED_TASKS_COL = 1         # 2nd value
CSV_FILENAME = "overall_failure_all_devices.csv"
# ---------------------

def extract_and_analyze_all_devices_failures():
    results = []

    print(f"--- 1. EXTRACTING FAILURE DATA (Iterations {ITERATION_START} to {ITERATION_END}) ---")
    
    for i in range(ITERATION_START, ITERATION_END + 1):
        ite_folder = os.path.join(BASE_DIR, f"ite{i}")
        
        if not os.path.exists(ite_folder):
            print(f"  -> Skipping iteration {i} - Directory '{ite_folder}' not found.")
            continue
            
        for policy, label in POLICIES.items():
            # Use a wildcard (*) to grab all device variations for this policy
            search_pattern = os.path.join(ite_folder, f"SIMRESULT_TWO_TIER_WITH_EO_{policy}_*DEVICES_{TARGET_APP}.log")
            log_files = glob.glob(search_pattern)
            
            for file_path in log_files:
                filename = os.path.basename(file_path)
                
                # Extract the device number from the filename using regex
                device_match = re.search(r'_(\d+)DEVICES_', filename)
                if not device_match:
                    continue
                devices = int(device_match.group(1))
                
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        
                        # Dynamically find the "Overall System" row
                        valid_lines = [line.strip() for line in lines if ';' in line and len(line.split(';')) > 5]
                        
                        if len(valid_lines) > 0:
                            stats_arr = valid_lines[0].split(';')
                            
                            completed_tasks = float(stats_arr[COMPLETED_TASKS_COL])
                            failed_tasks = float(stats_arr[FAILED_TASKS_COL])
                            
                            # Calculate Total Tasks and Failure Rate (%)
                            total_tasks = completed_tasks + failed_tasks
                            failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                            
                            results.append({
                                "Algorithm": label, 
                                "Devices": devices, 
                                "FailureRate": failure_rate
                            })
                        else:
                            print(f"  -> Error: No valid data rows found in {filename}")
                except Exception as e:
                    print(f"  -> Error reading {filename}: {e}")

    # Convert to DataFrame and Export
    if not results:
        print("\n❌ No data extracted. Please check your BASE_DIR and folders.")
        return

    df = pd.DataFrame(results)
    
    # Sort the CSV logically by Algorithm, then by Devices
    df = df.sort_values(by=["Algorithm", "Devices"])
    df.to_csv(CSV_FILENAME, index=False)
    
    print(f"✅ Extraction complete! Data saved to '{CSV_FILENAME}'.")
    print(f"Total data points collected: {len(df)}\n")

    print("--- 2. RUNNING STATISTICAL ANALYSIS (ALL DEVICES) ---")
    
    # Run ANOVA
    groups = [group['FailureRate'].values for name, group in df.groupby('Algorithm')]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("\n[ ANOVA RESULTS ]")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value:     {p_value:.10e}")
    
    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT variance detected across algorithms.\n")
    else:
        print("Result: NO significant variance detected (p >= 0.05).\n")

    # Run Tukey HSD
    print("[ TUKEY HSD (Head-to-Head) RESULTS ]")
    tukey = pairwise_tukeyhsd(endog=df['FailureRate'], groups=df['Algorithm'], alpha=0.05)
    print(tukey)
    
    # Print Summary Statistics
    print("\n[ SUMMARY STATISTICS (Failure Rate % - Averaged Across All Device Loads) ]")
    print(df.groupby('Algorithm')['FailureRate'].agg(['count', 'mean', 'std']))

if __name__ == "__main__":
    extract_and_analyze_all_devices_failures()