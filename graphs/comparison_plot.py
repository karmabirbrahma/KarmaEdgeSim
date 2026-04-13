import matplotlib.pyplot as plt
import numpy as np

metrics_time = ['Avg Service Time (s)', 'Avg Processing Time (s)']
baseline_time = [5.196, 4.438]   # ← Mobility OFF
on_time       = [5.139, 4.360]   # ← Mobility ON

metrics_failed = ['Failed Tasks']
baseline_failed = [10215]        # ← Mobility OFF
on_failed       = [8891]         # ← Mobility ON

# ---------------- Graph 1: Service & Processing Time ----------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(metrics_time))
width = 0.25

ax1.bar(x - width/2, baseline_time, width, label='Mobility OFF', color='lightblue')
ax1.bar(x + width/2, on_time, width, label='Mobility ON (HEO)', color='orange')

ax1.set_ylabel('Time (seconds)')
ax1.set_title('HEO: Service Time & Processing Time')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_time)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('SHEO_Time_Comparison.png')
plt.show()

# ---------------- Graph 2: Failed Tasks ----------------
fig2, ax2 = plt.subplots(figsize=(6, 5))
x2 = np.arange(len(metrics_failed))

ax2.bar(x2 - width/2, baseline_failed, width, label='Mobility OFF', color='lightblue')
ax2.bar(x2 + width/2, on_failed, width, label='Mobility ON (HEO)', color='orange')

ax2.set_ylabel('Number of Failed Tasks')
ax2.set_title('HEO: Failed Tasks Comparison')
ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_failed)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('SHEO_FailedTasks_Comparison.png')
plt.show()

print("   1. SHEO_Time_Comparison.png")
print("   2. SHEO_FailedTasks_Comparison.png")