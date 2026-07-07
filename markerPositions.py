import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the file
with open('processed_data/Opencap_Side/MarkerData/StaticTPose.trc', 'r') as f:
    lines = f.readlines()

# Extract marker names
marker_names_raw = lines[3].split('\t')
marker_names = [m.strip() for m in marker_names_raw if m.strip() not in ('Frame#', 'Time', '')]

# Extract first frame data
data_vals = [float(x) for x in lines[6].strip().split('\t') if x.strip() != '']
coords = data_vals[2:]

num_markers = len(marker_names)
points = np.array(coords[:num_markers*3]).reshape(-1, 3)

# Generate a wide figure containing all 3 orthographic views WITH labels
fig, axs = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('OpenCap Model Markers (Labeled) - Static T-Pose, Frame 1', fontsize=20, fontweight='bold')

# Helper function to add labels slightly offset from points to improve readability
def add_labels(ax, x_data, y_data, labels):
    for i, txt in enumerate(labels):
        ax.text(x_data[i] + 0.015, y_data[i], txt, fontsize=7, alpha=0.9, 
                verticalalignment='center')

# Front view (X vs Y)
axs[0].scatter(points[:, 0], points[:, 1], c='#0052cc', s=30)
add_labels(axs[0], points[:, 0], points[:, 1], marker_names)
axs[0].set_title('Side View', fontsize=16)
axs[0].set_xlabel('X (m) - Left/Right')
axs[0].set_ylabel('Y (m) - Up')
axs[0].axis('equal')
axs[0].grid(True, linestyle='--', alpha=0.6)

# Side view (Z vs Y)
axs[1].scatter(points[:, 2], points[:, 1], c='#d93025', s=30)
add_labels(axs[1], points[:, 2], points[:, 1], marker_names)
axs[1].set_title('Front View', fontsize=16)
axs[1].set_xlabel('Z (m) - Forward/Back')
axs[1].set_ylabel('Y (m) - Up')
axs[1].axis('equal')
axs[1].grid(True, linestyle='--', alpha=0.6)

# Top view (X vs Z)
axs[2].scatter(points[:, 0], points[:, 2], c='#0f9d58', s=30)
add_labels(axs[2], points[:, 0], points[:, 2], marker_names)
axs[2].set_title('Top View', fontsize=16)
axs[2].set_xlabel('X (m) - Left/Right')
axs[2].set_ylabel('Z (m) - Forward/Back')
axs[2].axis('equal')
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('SideCam_Markers.png', dpi=300)
print("Saved visuals")