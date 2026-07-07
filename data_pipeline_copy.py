import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import signal

def load_trc_files(file_path):
    """Loads raw TRC file and returns dataframe and framerate."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_rate = float(lines[2].split()[0])
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
    return df, data_rate

def calculate_time_offset(df_ref, df_target, marker_col, fps_ref):
    """Calculates the exact time offset between two dataframes."""
    dt = 1.0 / fps_ref
    t_ref_0 = df_ref['Time'].min()
    t_tgt_0 = df_target['Time'].min()
    
    t_ref_uniform = np.arange(0, df_ref['Time'].max() - t_ref_0, dt)
    t_tgt_uniform = np.arange(0, df_target['Time'].max() - t_tgt_0, dt)
    
    # Interpolate
    sig_ref = CubicSpline(df_ref['Time'], df_ref[marker_col])(t_ref_uniform + t_ref_0)
    sig_tgt = CubicSpline(df_target['Time'], df_target[marker_col])(t_tgt_uniform + t_tgt_0)
    
    # Normalize
    sig_ref = (sig_ref - np.mean(sig_ref)) / np.std(sig_ref)
    sig_tgt = (sig_tgt - np.mean(sig_tgt)) / np.std(sig_tgt)
    
    # Cross-Correlate
    correlation = signal.correlate(sig_ref, sig_tgt, mode='full')
    lags = signal.correlation_lags(len(sig_ref), len(sig_tgt), mode='full')
    best_lag = lags[np.argmax(correlation)]
    
    # Return total offset
    return (t_ref_0 - t_tgt_0) + (best_lag * dt)

def compare_recording_starts(mocap_path, side_path, front_path, trial_name, sync_marker="C7_Y"):
    print(f"\n--- Analyzing Click Sequence for {trial_name} ---")
    
    # 1. Load Data
    df_mocap, fps_mocap = load_trc_files(mocap_path)
    df_side, _ = load_trc_files(side_path)
    df_front, _ = load_trc_files(front_path)
    
    # Handle OpenCap naming convention for raw files
    opencap_marker = sync_marker if sync_marker in df_side.columns else f"{sync_marker.split('_')[0].lower()}_{sync_marker.split('_')[1]}"
    if opencap_marker not in df_side.columns:
        # Fallback if standard mapping fails
        opencap_marker = "C7_Y" 
        
    print(f"Syncing using MoCap {sync_marker} and OpenCap {opencap_marker}...")

    # 2. Calculate Offsets (Treating MoCap's physical start as absolute T=0)
    offset_side = calculate_time_offset(df_mocap, df_side, opencap_marker, fps_mocap)
    offset_front = calculate_time_offset(df_mocap, df_front, opencap_marker, fps_mocap)
    
    # Shift the times mathematically so their movements align
    df_mocap['Aligned_Time'] = df_mocap['Time'] - df_mocap['Time'].min()
    df_side['Aligned_Time'] = (df_side['Time'] - df_side['Time'].min()) + offset_side
    df_front['Aligned_Time'] = (df_front['Time'] - df_front['Time'].min()) + offset_front

    # 3. Determine the Click Sequence
    starts = {
        "MoCap": 0.0,
        "Side Camera": offset_side,
        "Front Camera": offset_front
    }
    
    # Sort from earliest (smallest number) to latest
    sorted_starts = sorted(starts.items(), key=lambda x: x[1])
    earliest_time = sorted_starts[0][1]
    
    print("\n📝 RECORDING TIMELINE REPORT:")
    for rank, (system, time_val) in enumerate(sorted_starts):
        delay_from_first = time_val - earliest_time
        print(f"  {rank + 1}. {system} clicked (Delay: +{delay_from_first:.2f} seconds)")

    # 4. Generate the Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot all three on a synchronized timeline
    plt.plot(df_mocap['Aligned_Time'], df_mocap[sync_marker], 
             label='MoCap', color='black', linewidth=3)
    plt.plot(df_side['Aligned_Time'], df_side[opencap_marker], 
             label='Side Camera', color='blue', linestyle='--', linewidth=2)
    plt.plot(df_front['Aligned_Time'], df_front[opencap_marker], 
             label='Front Camera', color='red', linestyle=':', linewidth=2)
    
    # Add vertical lines to show exactly where each recording started
    plt.axvline(x=starts["MoCap"], color='black', alpha=0.3, label="MoCap Start")
    plt.axvline(x=starts["Side Camera"], color='blue', alpha=0.3, label="Side Start")
    plt.axvline(x=starts["Front Camera"], color='red', alpha=0.3, label="Front Start")

    plt.title(f"Recording Start Sequence: {trial_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Absolute Unified Time (seconds)", fontsize=12)
    plt.ylabel(f"Position of {sync_marker}", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the plot
    os.makedirs("presentation_plots", exist_ok=True)
    save_path = f"presentation_plots/Click_Sequence_{trial_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n📊 Graph saved to {save_path}")

if __name__ == "__main__":
    # Test it on a specific trial
    trial = "SlowWalking4"
    
    mocap_file = f"raw_data/MoCap_Data/MarkerData/{trial}.trc"
    side_file = f"raw_data/Opencap_Side/MarkerData/{trial}.trc"
    front_file = f"raw_data/Opencap_Front/MarkerData/{trial}.trc"
    
    compare_recording_starts(mocap_file, side_file, front_file, trial, sync_marker="C7_Y")