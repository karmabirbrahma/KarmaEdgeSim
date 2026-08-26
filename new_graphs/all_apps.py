import os
import re
import glob
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_DIR = "../sim_results/ite30" # Ensure this points to your log folder
OUTPUT_DIR = "thesis_graphs/check" 

# Apps to process
APPS_TO_PLOT = [
    "ALL_APPS_GENERIC",
    "INFOTAINMENT_APP_GENERIC",
    "HEAVY_COMP_APP_GENERIC",
    "AUGMENTED_REALITY_GENERIC",
    "HEALTH_APP_GENERIC"
]

# 1. Filter to only these exactly 4 policies
POLICIES_TO_PLOT = ['SHO', 'PURE_DDQN', 'DDQN_MOB_ONLY', 'FUZZY_BASED']

# 2. Rename labels exactly as requested
LABELS = {
    'SHO': 'HEO', 
    'PURE_DDQN': 'DDQN', 
    'DDQN_MOB_ONLY': 'DDQN + Mobility', 
    'FUZZY_BASED': 'Fuzzy_Based'
}

# Styling
COLORS = {
    'HEO': '#2ca02c', 
    'PURE_DDQN': '#9467bd', 
    'DDQN_MOB_ONLY': '#8c564b', 
    'FUZZY_BASED': '#d62728'
}

MARKERS = {
    'HEO': 'o', 
    'PURE_DDQN': 'x', 
    'DDQN_MOB_ONLY': 'v', 
    'FUZZY_BASED': 's'
}

def main():
    print(f"Scanning directory: {TARGET_DIR}...")
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created new folder for graphs: {OUTPUT_DIR}/")

    for app_name in APPS_TO_PLOT:
        print(f"\nProcessing data for: {app_name}")
        process_and_plot_app(app_name)

def process_and_plot_app(app_name):
    file_pattern = os.path.join(TARGET_DIR, f"SIMRESULT_TWO_TIER_WITH_EO_*_*DEVICES_{app_name}.log")
    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print(f"  No logs found for {app_name}. Skipping.")
        return

    data = {}

    for file_path in log_files:
        filename = os.path.basename(file_path)
        
        match = re.search(r"SIMRESULT_TWO_TIER_WITH_EO_(.*)_(\d+)DEVICES_" + app_name + r"\.log", filename)
        if not match:
            continue
            
        policy = match.group(1)
        
        # Skip policies not in our filtered list
        if policy not in POLICIES_TO_PLOT:
            continue
            
        devices = int(match.group(2))
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                stats = lines[1].strip().split(';')
                
                # FIXED: Lowered threshold to 6 to stop skipping SHEO and DDQN
                if len(stats) >= 6: 
                    total_tasks = float(stats[0])
                    failed_tasks = float(stats[1])
                    
                    # FIXED INDICES: 
                    # stats[4] = Total Delay (which is Service Time)
                    # stats[5] = Processing Time
                    avg_delay = float(stats[4])
                    avg_service = float(stats[4]) 
                    avg_processing = float(stats[5]) 
                    
                    failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                    
                    if policy not in data:
                        data[policy] = {'devices': [], 'fails': [], 'delays': [], 'service': [], 'processing': []}
                        
                    data[policy]['devices'].append(devices)
                    data[policy]['fails'].append(failure_rate)
                    data[policy]['delays'].append(avg_delay)
                    data[policy]['service'].append(avg_service)
                    data[policy]['processing'].append(avg_processing)

    if not data:
        print(f"  No valid data parsed for {app_name}.")
        return

    # Sort data by device count so the lines draw correctly
    for policy in data:
        sorted_lists = sorted(zip(
            data[policy]['devices'], 
            data[policy]['fails'], 
            data[policy]['delays'], 
            data[policy]['service'], 
            data[policy]['processing']
        ))
        data[policy]['devices'] = [x[0] for x in sorted_lists]
        data[policy]['fails'] = [x[1] for x in sorted_lists]
        data[policy]['delays'] = [x[2] for x in sorted_lists]
        data[policy]['service'] = [x[3] for x in sorted_lists]
        data[policy]['processing'] = [x[4] for x in sorted_lists]

    display_title = app_name.replace('_GENERIC', '').replace('_', ' ').title()
    clean_filename = app_name.replace('_GENERIC', '')

    def plot_single_metric(metric_key, title_prefix, ylabel, file_suffix):
        plt.figure(figsize=(10, 7))
        
        for policy in POLICIES_TO_PLOT:
            if policy in data:
                metrics = data[policy]
                color = COLORS.get(policy, 'black')
                marker = MARKERS.get(policy, 'x')
                label = LABELS.get(policy, policy)
                
                linestyle = '-' if policy == 'SHO' else '--'
                linewidth = 4.0 if policy == 'SHO' else 2.5
                
                plt.plot(metrics['devices'], metrics[metric_key], label=label, color=color, 
                         marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=8)

        plt.title(f'{title_prefix} - {display_title}', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Mobile Devices', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        output_filename = f"{file_suffix}_{clean_filename}.png"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"    -> Saved {title_prefix} graph: {save_path}")

    # Plot the 4 separate metrics
    plot_single_metric('delays', 'Average Delay', 'Average Total Delay (Seconds)', 'Delay')
    plot_single_metric('fails', 'Failure Rate', 'Failed Tasks (%)', 'FailureRate')
    plot_single_metric('service', 'Average Service Time', 'Service Time (Seconds)', 'ServiceTime')
    plot_single_metric('processing', 'Average Processing Time', 'Processing Time (Seconds)', 'ProcessingTime')

if __name__ == "__main__":
    main()