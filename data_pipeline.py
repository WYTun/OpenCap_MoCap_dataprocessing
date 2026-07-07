import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def load_trc_files(file_path):
    """Utility function to load the raw TRC files for the 'Before' plots."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    raw_markers = lines[3].strip().split('\t')
    markers = [m.strip() for m in raw_markers if m.strip() != '']

    clean_columns = ['Frame#', 'Time']
    for marker in markers[2:]:
        clean_columns.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    df = pd.read_csv(file_path, skiprows=5, sep='\t', header=None) 
    df = df.dropna(axis=1, how='all')
    df = df.iloc[:, :len(clean_columns)]
    df.columns = clean_columns
    df = df.dropna(subset=['Time'])
    return df

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

if __name__ == "__main__":
    # Define directories
    raw_mocap_dir = "raw_data/MoCap_Data/MarkerData/"
    raw_opencap_dir = "raw_data/Opencap_Front/MarkerData/"
    processed_dir = "processed_data/"
    output_images_dir = "presentation_images/"
    
    # Marker to visualize (C7_Y is usually the best indicator of sync success)
    marker_to_plot = "RKNE_Z"
    
    print("Scanning for processed trials to plot...\n")
    
    # Find all the raw TRC files
    search_pattern = os.path.join(raw_opencap_dir, "*.trc")
    raw_side_files = glob.glob(search_pattern)
    
    for raw_opencap_path in raw_side_files:
        filename = os.path.basename(raw_opencap_path)
        base_name = filename.replace(".trc", "")
        
        # Define paths for the 3 other required files
        raw_mocap_path = os.path.join(raw_mocap_dir, filename)
        clean_mocap_path = os.path.join(processed_dir, f"aligned_{base_name}_mocap_cleaned.csv")
        clean_opencap_path = os.path.join(processed_dir, f"aligned_{base_name}_opencap_side_resampled.csv")
        
        # Check if all files exist (meaning the pipeline succeeded for this trial)
        if not all(os.path.exists(p) for p in [raw_mocap_path, clean_mocap_path, clean_opencap_path]):
            print(f"[SKIP] Missing processed or raw files for {base_name}. It may have failed in processing.")
            continue
            
        print(f"Plotting {base_name}...")
        
        # Load the data
        df_mocap_raw = load_trc_files(raw_mocap_path)
        df_opencap_raw = load_trc_files(raw_opencap_path)
        df_mocap_clean = pd.read_csv(clean_mocap_path)
        df_opencap_clean = pd.read_csv(clean_opencap_path)
        
        # Generate the presentation graphic
        generate_presentation_plot(
            mocap_raw=df_mocap_raw,
            opencap_raw=df_opencap_raw,
            mocap_clean=df_mocap_clean,
            opencap_clean=df_opencap_clean,
            marker_col=marker_to_plot,
            trial_name=base_name,
            output_folder=output_images_dir
        )
        
    print("\nAll presentation plots generated successfully!")