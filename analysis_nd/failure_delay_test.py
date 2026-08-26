import os
import glob
import re
import pandas as pd
import scipy.stats as stats

# --- CONFIGURATION ---
BASE_DIR = "../sim_results" 
ITERATION_START = 31
ITERATION_END = 45  

TARGET_APP = "ALL_APPS_GENERIC"

# We only need HEO and Pure DDQN for the direct t-test
POLICIES = {
    'SHO': 'HEO', 
    'PURE_DDQN': 'DDQN'
}

COMPLETED_TASKS_COL = 0
FAILED_TASKS_COL = 1
DELAY_COL = 4         
CSV_FILENAME = "heo_vs_ddqn_direct_test.csv"
# ---------------------

def extract_and_test():
    results = []
    
    print(f"--- 1. EXTRACTING DATA (Iterations {ITERATION_START} to {ITERATION_END}) ---")
    
    for i in range(ITERATION_START, ITERATION_END + 1):
        ite_folder = os.path.join(BASE_DIR, f"ite{i}")
        
        if not os.path.exists(ite_folder):
            print(f"  -> Skipping iteration {i} - Directory '{ite_folder}' not found.")
            continue
            
        for policy, label in POLICIES.items():
            search_pattern = os.path.join(ite_folder, f"SIMRESULT_TWO_TIER_WITH_EO_{policy}_*DEVICES_{TARGET_APP}.log")
            log_files = glob.glob(search_pattern)
            
            for file_path in log_files:
                filename = os.path.basename(file_path)
                
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
                            
                            # 1. Extract Failure Data
                            completed_tasks = float(stats_arr[COMPLETED_TASKS_COL])
                            failed_tasks = float(stats_arr[FAILED_TASKS_COL])
                            total_tasks = completed_tasks + failed_tasks
                            failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                            
                            # 2. Extract Delay Data
                            delay_val = float(stats_arr[DELAY_COL])
                            
                            results.append({
                                "Algorithm": label, 
                                "Devices": devices, 
                                "Delay": delay_val,
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
    df = df.sort_values(by=["Algorithm", "Devices"])
    df.to_csv(CSV_FILENAME, index=False)
    
    print(f"✅ Extraction complete! Data saved to '{CSV_FILENAME}'.")
    print(f"Total data points collected: {len(df)}\n")

    print("--- 2. INDEPENDENT T-TEST RESULTS (HEO vs. PURE DDQN) ---")
    
    # Isolate the data for each algorithm
    heo_data = df[df['Algorithm'] == 'HEO']
    ddqn_data = df[df['Algorithm'] == 'DDQN']
    
    # 1. Perform T-Test for Average Total Delay
    t_stat_delay, p_val_delay = stats.ttest_ind(heo_data['Delay'], ddqn_data['Delay'])
    
    print("\n[ 1. AVERAGE TOTAL DELAY ]")
    print(f"HEO Mean:  {heo_data['Delay'].mean():.6f} seconds")
    print(f"DDQN Mean: {ddqn_data['Delay'].mean():.6f} seconds")
    print(f"t-statistic: {t_stat_delay:.4f}")
    print(f"p-value:     {p_val_delay:.6f}")
    
    if p_val_delay < 0.05:
        print("-> Conclusion: STATISTICALLY SIGNIFICANT difference in delay.")
    else:
        print("-> Conclusion: NO STATISTICALLY SIGNIFICANT difference (Statistically Equivalent).")

    # 2. Perform T-Test for Failure Rate
    t_stat_fail, p_val_fail = stats.ttest_ind(heo_data['FailureRate'], ddqn_data['FailureRate'])
    
    print("\n[ 2. FAILURE RATE % ]")
    print(f"HEO Mean:  {heo_data['FailureRate'].mean():.6f} %")
    print(f"DDQN Mean: {ddqn_data['FailureRate'].mean():.6f} %")
    print(f"t-statistic: {t_stat_fail:.4f}")
    print(f"p-value:     {p_val_fail:.6f}")
    
    if p_val_fail < 0.05:
        print("-> Conclusion: STATISTICALLY SIGNIFICANT difference in failure rate.")
    else:
        print("-> Conclusion: NO STATISTICALLY SIGNIFICANT difference (Statistically Equivalent).")

if __name__ == "__main__":
    extract_and_test()
