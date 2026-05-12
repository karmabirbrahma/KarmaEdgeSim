import os
import re
import glob
import matplotlib.pyplot as plt

# Define the target directory containing the logs
TARGET_DIR = "../sim_results/ite13"

def main():
    print(f"Scanning directory: {TARGET_DIR}...")
    
    # Regex to extract the Policy and Number of Devices from the filename
    # Example: SIMRESULT_TWO_TIER_WITH_EO_DDQN_200DEVICES_ALL_APPS_GENERIC.log
    file_pattern = os.path.join(TARGET_DIR, "SIMRESULT_TWO_TIER_WITH_EO_*_*DEVICES_ALL_APPS_GENERIC.log")
    log_files = glob.glob(file_pattern)
    
    if not log_files:
        print(f"Error: No ALL_APPS_GENERIC.log files found in {TARGET_DIR}.")
        print("Make sure you are running this script from the root EdgeCloudSim-DeepEdge folder.")
        return

    # Dictionary to store the extracted data: data[policy][devices] = (failure_rate, avg_delay)
    data = {}

    for file_path in log_files:
        filename = os.path.basename(file_path)
        
        # Parse filename to get Policy and Devices
        match = re.search(r"SIMRESULT_TWO_TIER_WITH_EO_(.*)_(\d+)DEVICES_ALL_APPS_GENERIC\.log", filename)
        if not match:
            continue
            
        policy = match.group(1)
        devices = int(match.group(2))
        
        # Read the log file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Line 0 is the auto-generated comment, Line 1 contains the actual data
            if len(lines) > 1:
                stats = lines[1].strip().split(';')
                if len(stats) >= 5:
                    total_tasks = float(stats[0])
                    failed_tasks = float(stats[1])
                    avg_delay = float(stats[4])
                    
                    # Calculate actual failure percentage
                    failure_rate = (failed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
                    
                    if policy not in data:
                        data[policy] = {'devices': [], 'fails': [], 'delays': []}
                        
                    data[policy]['devices'].append(devices)
                    data[policy]['fails'].append(failure_rate)
                    data[policy]['delays'].append(avg_delay)

    # Sort the data by device count for smooth plotting
    for policy in data:
        # Zip, sort, and unzip the lists based on device count
        sorted_lists = sorted(zip(data[policy]['devices'], data[policy]['fails'], data[policy]['delays']))
        data[policy]['devices'] = [x[0] for x in sorted_lists]
        data[policy]['fails'] = [x[1] for x in sorted_lists]
        data[policy]['delays'] = [x[2] for x in sorted_lists]

    print(f"Successfully extracted data for {len(data)} policies.")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Styling configurations
    colors = {'DDQN': '#2ca02c', 'FUZZY_COMPETITOR': '#d62728', 'NETWORK_BASED': '#1f77b4', 'UTILIZATION_BASED': '#ff7f0e'}
    markers = {'DDQN': 'o', 'FUZZY_COMPETITOR': 's', 'NETWORK_BASED': '^', 'UTILIZATION_BASED': 'D'}
    labels = {'DDQN': 'S-HEO (Our DDQN AI)', 'FUZZY_COMPETITOR': 'Fuzzy Baseline', 
              'NETWORK_BASED': 'Network Baseline', 'UTILIZATION_BASED': 'Utilization Baseline'}

    for policy, metrics in data.items():
        color = colors.get(policy, 'black')
        marker = markers.get(policy, 'x')
        label = labels.get(policy, policy)
        
        linestyle = '-' if policy == 'DDQN' else '--'
        linewidth = 3 if policy == 'DDQN' else 2
        
        # Plot Delay
        ax1.plot(metrics['devices'], metrics['delays'], label=label, color=color, 
                 marker=marker, linewidth=linewidth, markersize=8)
        
        # Plot Failures
        ax2.plot(metrics['devices'], metrics['fails'], label=label, color=color, 
                 marker=marker, linewidth=linewidth, markersize=8)

    # Configure Delay Subplot
    ax1.set_title('Average Total Delay vs Network Congestion', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Mobile Devices', fontsize=12)
    ax1.set_ylabel('Average Delay (Seconds)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=12)

    # Configure Failure Subplot
    ax2.set_title('Task Failure Rate vs Network Congestion', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Mobile Devices', fontsize=12)
    ax2.set_ylabel('Failed Tasks (%)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    
    # Save the graph
    output_filename = "ite13_final_results.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n🎉 Graph generated successfully! Saved as: {output_filename}")

if __name__ == "__main__":
    main()