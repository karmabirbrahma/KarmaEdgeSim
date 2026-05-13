import os
import re
import glob
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_DIR = "../sim_results/ite20" # Change this to the folder containing your final logs

# NEW: The folder where you want to save the generated graphs
OUTPUT_DIR = "thesis_graphs/ite20" 

# Added _GENERIC to all files to match EdgeCloudSim's exact naming format
APPS_TO_PLOT = [
    "ALL_APPS_GENERIC",
    "INFOTAINMENT_APP_GENERIC",
    "HEAVY_COMP_APP_GENERIC",
    "AUGMENTED_REALITY_GENERIC",
    "HEALTH_APP_GENERIC"
]

def main():
    print(f"Scanning directory: {TARGET_DIR}...")
    
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created new folder for graphs: {OUTPUT_DIR}/")

    for app_name in APPS_TO_PLOT:
        print(f"\nProcessing data for: {app_name}")
        process_and_plot_app(app_name)

def process_and_plot_app(app_name):
    # Find all logs for this specific application
    file_pattern = os.path.join(TARGET_DIR, f"SIMRESULT_TWO_TIER_WITH_EO_*_*DEVICES_{app_name}.log")
    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print(f"  No logs found for {app_name}. Skipping.")
        return

    data = {}

    for file_path in log_files:
        filename = os.path.basename(file_path)
        
        # Parse filename. E.g.: SIMRESULT_TWO_TIER_WITH_EO_SHO_2400DEVICES_HEAVY_COMP_APP_GENERIC.log
        match = re.search(r"SIMRESULT_TWO_TIER_WITH_EO_(.*)_(\d+)DEVICES_" + app_name + r"\.log", filename)
        if not match:
            continue
            
        policy = match.group(1)
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

    if not data:
        print(f"  No valid data parsed for {app_name}.")
        return

    # Sort data by device count
    for policy in data:
        sorted_lists = sorted(zip(data[policy]['devices'], data[policy]['fails'], data[policy]['delays']))
        data[policy]['devices'] = [x[0] for x in sorted_lists]
        data[policy]['fails'] = [x[1] for x in sorted_lists]
        data[policy]['delays'] = [x[2] for x in sorted_lists]

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    colors = {
        'SHO': '#2ca02c', 'PURE_DDQN': '#9467bd', 'DDQN_MOB_ONLY': '#8c564b', 
        'FUZZY_COMPETITOR': '#d62728', 'NETWORK_BASED': '#1f77b4', 
        'UTILIZATION_BASED': '#ff7f0e', 'HYBRID': '#e377c2'
    }
    markers = {
        'SHO': 'o', 'PURE_DDQN': 'x', 'DDQN_MOB_ONLY': 'v', 
        'FUZZY_COMPETITOR': 's', 'NETWORK_BASED': '^', 
        'UTILIZATION_BASED': 'D', 'HYBRID': '*'
    }
    labels = {
        'SHO': 'S-HEO (Our Full AI)', 'PURE_DDQN': 'Pure DDQN', 
        'DDQN_MOB_ONLY': 'DDQN + Mobility', 'FUZZY_COMPETITOR': 'Fuzzy', 
        'NETWORK_BASED': 'Network', 'UTILIZATION_BASED': 'Utilization',
        'HYBRID': 'Hybrid'
    }

    # Clean up the title (Removes _GENERIC and formats nicely)
    display_title = app_name.replace('_GENERIC', '').replace('_', ' ').title()

    for policy, metrics in data.items():
        color = colors.get(policy, 'black')
        marker = markers.get(policy, 'x')
        label = labels.get(policy, policy)
        
        linestyle = '-' if policy == 'SHO' else '--'
        linewidth = 3.5 if policy == 'SHO' else 2.0
        
        ax1.plot(metrics['devices'], metrics['delays'], label=label, color=color, 
                 marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)
        ax2.plot(metrics['devices'], metrics['fails'], label=label, color=color, 
                 marker=marker, linewidth=linewidth, linestyle=linestyle, markersize=7)

    # Delay Plot Styling
    ax1.set_title(f'Avg Delay - {display_title}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Mobile Devices', fontsize=12)
    ax1.set_ylabel('Average Total Delay (Seconds)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=10)

    # Failure Plot Styling
    ax2.set_title(f'Failure Rate - {display_title}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Mobile Devices', fontsize=12)
    ax2.set_ylabel('Failed Tasks (%)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    
    # Clean output filename and attach the target directory path
    clean_filename = app_name.replace('_GENERIC', '')
    output_filename = f"thesis_results_{clean_filename}.png"
    
    # Save inside the new directory!
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"  -> Saved graph as: {save_path}")

if __name__ == "__main__":
    main()