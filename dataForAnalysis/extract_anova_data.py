import os
import glob
import re
import pandas as pd

# --- CONFIGURATION ---
# Base directory where your iteration folders are located
BASE_DIR = "../sim_results/ite49" 

# Target device load for the ANOVA (e.g., peak load where variance is highest)
TARGET_DEVICES = "2400" 
TARGET_APP = "ALL_APPS_GENERIC"

# Map the exact policy names from your files to the clean labels for the CSV
POLICIES = {
    'SHO': 'HEO', 
    'PURE_DDQN': 'DDQN', 
    'DDQN_MOB_ONLY': 'DDQN_Mobility', 
    'FUZZY_BASED': 'Fuzzy_Based'
}

# Based on your plotting script:
TARGET_ROW = 1  # 2nd line of the log file
TARGET_COL = 4  # 5th value separated by ';' (Average Total Delay)
# ---------------------

def extract_anova_data():
    results = []

    # Loop through iterations 1 to 10 (or up to 30 if you ran more)
    for i in range(1, 2):
        ite_folder = BASE_DIR
        
        if not os.path.exists(ite_folder):
            print(f"Skipping iteration {i} - Directory '{ite_folder}' not found.")
            continue
            
        print(f"Scanning {ite_folder}...")
        
        for policy, label in POLICIES.items():
            # Construct the exact filename based on your plotting script's logic
            expected_filename = f"SIMRESULT_TWO_TIER_WITH_EO_{policy}_{TARGET_DEVICES}DEVICES_{TARGET_APP}.log"
            file_path = os.path.join(ite_folder, expected_filename)
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > TARGET_ROW:
                            stats = lines[TARGET_ROW].strip().split(';')
                            if len(stats) > TARGET_COL:
                                delay_val = float(stats[TARGET_COL])
                                results.append({"Algorithm": label, "Delay": delay_val})
                            else:
                                print(f"  -> Error: Column {TARGET_COL} missing in {expected_filename}")
                        else:
                            print(f"  -> Error: Row {TARGET_ROW} missing in {expected_filename}")
                except Exception as e:
                    print(f"  -> Error reading {expected_filename}: {e}")
            else:
                print(f"  -> Missing file: {expected_filename}")

    # Convert to DataFrame and Export
    if results:
        df = pd.DataFrame(results)
        df.to_csv("delay_results_19.csv", index=False)
        print("\n✅ Extraction complete! Data saved to 'delay_results_19.csv'.")
        print(f"Total data points collected: {len(df)}")
        print("\nPreview:")
        print(df.head())
    else:
        print("\n❌ No data extracted. Please check your BASE_DIR and folder structure.")

if __name__ == "__main__":
    extract_anova_data()
