import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Marker mapping for translation
MARKER_MAPPING = {
    "C7": "C7", "sternum": "STRN", 
    "r_ASIS": "RASI", "l_ASIS": "LASI", 
    "r_PSIS": "RPSI", "l_PSIS": "LPSI",
    "r_shoulder": "RSHO", "l_shoulder": "LSHO",
    "r_elbow": "RELB", "l_elbow": "LELB",
    "r_wrist_radius": "RWRA", "l_wrist_radius": "LWRA",
    "r_wrist_ulna": "RWRB", "l_wrist_ulna": "LWRB",
    "r_index": "RFIN", "l_index": "LFIN",
    "r_knee": "RKNE", "l_knee": "LKNE",
    "r_ankle": "RANK", "l_ankle": "LANK",
    "r_calc": "RHEE", "l_calc": "LHEE",
    "r_toe": "RTOE", "l_toe": "LTOE"
}

def load_trc_files(file_path):
    """Utility function to load raw TRC files."""
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

def generate_3way_presentation_plot(mocap_raw, side_raw, front_raw, 
                                    mocap_clean, side_clean, front_clean, 
                                    target_mocap_marker, trial_name, output_folder, mapping_dict):
    
    # 1. Reverse translate the marker name for the Raw OpenCap files
    base_mocap = target_mocap_marker.rsplit('_', 1)[0]
    axis = target_mocap_marker.rsplit('_', 1)[1]
    
    reverse_mapping = {v: k for k, v in mapping_dict.items()}
    base_opencap = reverse_mapping.get(base_mocap, base_mocap)
    raw_opencap_marker = f"{base_opencap}_{axis}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------------------------------------------------
    # PLOT 1: BEFORE (Raw Data Shifted to T=0)
    # ---------------------------------------------------------
    t0_mocap = mocap_raw['Time'] - mocap_raw['Time'].min()
    t0_side = side_raw['Time'] - side_raw['Time'].min()
    t0_front = front_raw['Time'] - front_raw['Time'].min()
    
    axes[0].plot(t0_mocap, mocap_raw[target_mocap_marker], 
                 label='Raw MoCap (~100Hz)', color='black', linewidth=2.5, alpha=0.7)
    
    if raw_opencap_marker in side_raw.columns:
        axes[0].plot(t0_side, side_raw[raw_opencap_marker], 
                     label='Raw Side Camera (60Hz)', color='blue', linestyle='--', linewidth=2)
        axes[0].plot(t0_front, front_raw[raw_opencap_marker], 
                     label='Raw Front Camera (60Hz)', color='red', linestyle=':', linewidth=2.5)
    else:
        print(f"    [Warning] Could not find {raw_opencap_marker} in raw OpenCap data!")
    
    axes[0].set_title(f"BEFORE: Unsynced Recording Delays\n({trial_name})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time from start of recording (s)", fontsize=12)
    axes[0].set_ylabel("Position (meters)", fontsize=12)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # ---------------------------------------------------------
    # PLOT 2: AFTER (Processed Data)
    # ---------------------------------------------------------
    axes[1].plot(mocap_clean['Time'], mocap_clean[target_mocap_marker], 
                 label='Processed MoCap', color='black', linewidth=2.5, alpha=0.7)
    axes[1].plot(side_clean['Time'], side_clean[target_mocap_marker], 
                 label='Synced Side Camera', color='blue', linestyle='--', linewidth=2)
    axes[1].plot(front_clean['Time'], front_clean[target_mocap_marker], 
                 label='Synced Front Camera', color='red', linestyle=':', linewidth=2.5)
    
    axes[1].set_title(f"AFTER: Cross-Correlated & Aligned\n({trial_name})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Absolute Aligned Time (s)", fontsize=12)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Format axes to ensure fair visual comparison
    try:
        y_min = min(mocap_clean[target_mocap_marker].min(), side_clean[target_mocap_marker].min(), front_clean[target_mocap_marker].min())
        y_max = max(mocap_clean[target_mocap_marker].max(), side_clean[target_mocap_marker].max(), front_clean[target_mocap_marker].max())
        padding = (y_max - y_min) * 0.15
        axes[0].set_ylim(y_min - padding, y_max + padding)
        axes[1].set_ylim(y_min - padding, y_max + padding)
        
        # Zoom into the first 3 seconds to highlight the delay
        axes[0].set_xlim(0, 3) 
    except:
        pass 

    # ---------------------------------------------------------
    # FORMAT AND SAVE
    # ---------------------------------------------------------
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, f"Slide_Plot_3Way_{trial_name}.png")
    
    plt.savefig(save_path, dpi=300) 
    plt.close()
    print(f"  -> Generated: {save_path}")

if __name__ == "__main__":
    # Define directories
    raw_mocap_dir = "raw_data/MoCap_Data/MarkerData/"
    raw_side_dir = "raw_data/Opencap_Side/MarkerData/"
    raw_front_dir = "raw_data/Opencap_Front/MarkerData/"
    processed_dir = "processed_data/"
    output_images_dir = "presentation_plots/"
    
    # Choose your marker
    marker_to_plot = "C7_Y"  # You can change this to any marker you want to visualize
    
    print("Scanning for processed trials to plot...\n")
    
    search_pattern = os.path.join(raw_side_dir, "*.trc")
    raw_side_files = glob.glob(search_pattern)
    
    for side_path in raw_side_files:
        filename = os.path.basename(side_path)
        base_name = filename.replace(".trc", "")
        
        # Build paths for all 6 required files
        raw_mocap_path = os.path.join(raw_mocap_dir, filename)
        raw_front_path = os.path.join(raw_front_dir, filename)
        
        clean_mocap_path = os.path.join(processed_dir, f"aligned_{base_name}_mocap_cleaned.csv")
        clean_side_path = os.path.join(processed_dir, f"aligned_{base_name}_opencap_side_resampled.csv")
        clean_front_path = os.path.join(processed_dir, f"aligned_{base_name}_opencap_front_resampled.csv")
        
        # Check if all files exist
        required_files = [raw_mocap_path, side_path, raw_front_path, clean_mocap_path, clean_side_path, clean_front_path]
        if not all(os.path.exists(p) for p in required_files):
            print(f"[SKIP] Missing files for {base_name}. Pipeline may have skipped it.")
            continue
            
        print(f"Plotting 3-way graph for {base_name}...")
        
        # Load the data
        df_mocap_raw = load_trc_files(raw_mocap_path)
        df_side_raw = load_trc_files(side_path)
        df_front_raw = load_trc_files(raw_front_path)
        
        df_mocap_clean = pd.read_csv(clean_mocap_path)
        df_side_clean = pd.read_csv(clean_side_path)
        df_front_clean = pd.read_csv(clean_front_path)
        
        # Generate the graphic
        generate_3way_presentation_plot(
            mocap_raw=df_mocap_raw,
            side_raw=df_side_raw,
            front_raw=df_front_raw,
            mocap_clean=df_mocap_clean,
            side_clean=df_side_clean,
            front_clean=df_front_clean,
            target_mocap_marker=marker_to_plot,
            trial_name=base_name,
            output_folder=output_images_dir,
            mapping_dict=MARKER_MAPPING
        )
        
    print("\nAll 3-way presentation plots generated successfully!")