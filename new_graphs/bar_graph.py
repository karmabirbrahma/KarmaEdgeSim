import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_DIR = "../sim_results/ite21" # Update to your latest results folder
OUTPUT_DIR = "thesis_graphs/done" 

APP_TO_PLOT = "ALL_APPS_GENERIC" # We will use the overall average for the bar chart

# Only comparing the two main contenders
POLICIES_TO_PLOT = ['SHO', 'PURE_DDQN']

def main():
    print(f"Scanning directory: {TARGET_DIR}...")
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    file_pattern = os.path.join(TARGET_DIR, f"SIMRESULT_TWO_TIER_WITH_EO_*_*DEVICES_{APP_TO_PLOT}.log")
    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print(f"No logs found for {APP_TO_PLOT}. Skipping.")
        return

    data = {}

    # 1. Extract Data
    for file_path in log_files:
        filename = os.path.basename(file_path)
        match = re.search(r"SIMRESULT_TWO_TIER_WITH_EO_(.*)_(\d+)DEVICES_" + APP_TO_PLOT + r"\.log", filename)
        if not match:
            continue
            
        policy = match.group(1)
        if policy not in POLICIES_TO_PLOT:
            continue
            
        devices = int(match.group(2))
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                stats = lines[1].strip().split(';')
                if len(stats) >= 5:
                    total_tasks = float(stats[0])
                    failed_tasks = float(stats[1])
                    avg_delay = float(stats[4])
                    
                    failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                    
                    if policy not in data:
                        data[policy] = {'devices': [], 'fails': [], 'delays': []}
                        
                    data[policy]['devices'].append(devices)
                    data[policy]['fails'].append(failure_rate)
                    data[policy]['delays'].append(avg_delay)

    # 2. Sort Data by Devices
    for policy in data:
        sorted_lists = sorted(zip(data[policy]['devices'], data[policy]['fails'], data[policy]['delays']))
        data[policy]['devices'] = [x[0] for x in sorted_lists]
        data[policy]['fails'] = [x[1] for x in sorted_lists]
        data[policy]['delays'] = [x[2] for x in sorted_lists]

    # Ensure both policies have the same device counts for the bar chart
    if 'SHO' not in data or 'PURE_DDQN' not in data:
        print("Missing data for one of the policies!")
        return
        
    devices = data['SHO']['devices']
    
    # 3. Setup Bar Chart Parameters
    x = np.arange(len(devices))  # the label locations
    width = 0.35  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- PLOT 1: DELAY ---
    rects1 = ax1.bar(x - width/2, data['SHO']['delays'], width, label='HEO', color='#2ca02c', edgecolor='black')
    rects2 = ax1.bar(x + width/2, data['PURE_DDQN']['delays'], width, label='Pure DDQN', color='#9467bd', edgecolor='black')

    ax1.set_title('Average Delay Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Mobile Devices', fontsize=14)
    ax1.set_ylabel('Average Total Delay (Seconds)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(devices)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)

    # --- PLOT 2: FAILURE RATE ---
    rects3 = ax2.bar(x - width/2, data['SHO']['fails'], width, label='HEO', color='#2ca02c', edgecolor='black')
    rects4 = ax2.bar(x + width/2, data['PURE_DDQN']['fails'], width, label='Pure DDQN', color='#9467bd', edgecolor='black')

    ax2.set_title('Task Failure Rate Comparison', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Number of Mobile Devices', fontsize=14)
    ax2.set_ylabel('Failed Tasks (%)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(devices)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)

    # Optional: Add value labels on top of the bars
    def autolabel(rects, ax, is_percent=False):
        for rect in rects:
            height = rect.get_height()
            label_text = f'{height:.2f}%' if is_percent else f'{height:.2f}s'
            # Only add labels to bars that are tall enough to matter visually
            if height > 0.01: 
                ax.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, rotation=45)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax2, is_percent=True)
    autolabel(rects4, ax2, is_percent=True)

    plt.tight_layout()
    
    # Save the file
    output_filename = "HEO_vs_DDQN.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Bar chart generated successfully! Saved as: {save_path}")

if __name__ == "__main__":
    main()