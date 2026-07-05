import pandas as pd
import matplotlib.pyplot as plt
import os
from data_pipeline import load_trc_files

def generate_presentation_plot(mocap_raw, opencap_raw, mocap_clean, opencap_clean, marker_col, trial_name, output_folder):
    """Generates a high-resolution, side-by-side 'Before and After' plot."""
    
    # Set up a wide, side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------------------------------------------------
    # PLOT 1: BEFORE (Raw Data)
    # ---------------------------------------------------------
    # Note: If OpenCap raw data looks wildly different in height, it highlights 
    # the coordinate mismatch that your pipeline successfully fixes!
    axes[0].plot(mocap_raw['Time'], mocap_raw[marker_col], 
                 label='Raw MoCap (~100Hz)', color='black', linewidth=2.5, alpha=0.7)
    
    # We use a try-except block just in case the raw OpenCap file has slightly 
    # different column names before your mapping function standardizes them.
    opencap_marker = marker_col if marker_col in opencap_raw.columns else f"{marker_col.split('_')[0]}_Y"
    if opencap_marker in opencap_raw.columns:
        axes[0].plot(opencap_raw['Time'], opencap_raw[opencap_marker], 
                     label='Raw OpenCap (60Hz)', color='red', linestyle='--', linewidth=2)
    
    axes[0].set_title(f"BEFORE: Raw Unsynced Data\n({trial_name} - {marker_col})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Original Time (seconds)", fontsize=12)
    axes[0].set_ylabel("Position (meters)", fontsize=12)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # ---------------------------------------------------------
    # PLOT 2: AFTER (Processed Data)
    # ---------------------------------------------------------
    axes[1].plot(mocap_clean['Time'], mocap_clean[marker_col], 
                 label='Processed MoCap', color='black', linewidth=2.5, alpha=0.7)
    axes[1].plot(opencap_clean['Time'], opencap_clean[marker_col], 
                 label='Synced & Resampled OpenCap', color='red', linestyle='--', linewidth=2)
    
    axes[1].set_title(f"AFTER: Time-Synced & FPS Aligned\n({trial_name} - {marker_col})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Aligned Time (seconds)", fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Format axes to ensure fair visual comparison
    try:
        y_min = min(mocap_clean[marker_col].min(), opencap_clean[marker_col].min())
        y_max = max(mocap_clean[marker_col].max(), opencap_clean[marker_col].max())
        padding = (y_max - y_min) * 0.15
        axes[0].set_ylim(y_min - padding, y_max + padding)
        axes[1].set_ylim(y_min - padding, y_max + padding)
    except:
        pass # Fallback to auto-scaling if an error occurs calculating bounds

    # ---------------------------------------------------------
    # FORMAT AND SAVE
    # ---------------------------------------------------------
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"Slide_Plot_{trial_name}.png")
    
    plt.savefig(save_path, dpi=300) # High-res for presentations
    plt.close()
    print(f"  -> Generated: {save_path}")