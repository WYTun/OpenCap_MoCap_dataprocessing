import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import glob

#mapping MoCap and OpenCap markers
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
    """
    Reads a OpenSim .trc marker file, skips the metadata headers,
    and returns a clean pandas DataFrame and its original frame rate.
    """   
    #read data rate from the file from 3rd line
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_rate = float(lines[2].split()[0])
    raw_markers = lines[3].strip().split('\t')
    markers = [m.strip() for m in raw_markers if m.strip() != '']

    #fix column names
    clean_columns = ['Frame#', 'Time']
    for marker in markers[2:]:
        clean_columns.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

    #skip all meta data and col names 
    df = pd.read_csv(file_path, skiprows=5, sep='\t', header=None) 
    df = df.dropna(axis=1, how='all')

    #reattach the fixed column name to df
    df = df.iloc[:, :len(clean_columns)]
    df.columns = clean_columns

    df = df.dropna(subset=['Time']) #drop rows without time

    return df, data_rate

def align_columns(df_opencap, df_mocap, marker_dict):
    print("  -> Aligning OpenCap column names and filtering extra markers...")
    
    full_column_mapping = {}
    for opencap_name, mocap_name in marker_dict.items():
        full_column_mapping[f"{opencap_name}_X"] = f"{mocap_name}_X"
        full_column_mapping[f"{opencap_name}_Y"] = f"{mocap_name}_Y"
        full_column_mapping[f"{opencap_name}_Z"] = f"{mocap_name}_Z"

    df_opencap_renamed = df_opencap.rename(columns=full_column_mapping)

    mocap_columns = set(df_mocap.columns)
    opencap_columns = set(df_opencap_renamed.columns)

    columns_to_keep = ['Time'] + [col for col in df_mocap.columns if col in opencap_columns and col != 'Time']
    df_aligned = df_opencap_renamed[columns_to_keep]

    return df_aligned

def process_and_align_trajectories(opencap_side_path,opencap_front_path,mocap_path, output_path):
    """
    Loads both files, extracts overlapping times, interpolates 
    OpenCap (60Hz) to match MoCap (100.092Hz), and saves the result.
    """

    #load raw files
    df_opencap_side, fps_opencap_side = load_trc_files(opencap_side_path)
    df_opencap_front, fps_opencap_front = load_trc_files(opencap_front_path)
    df_mocap, fps_mocap = load_trc_files(mocap_path)

    #filter and align OpenCap columns to match MoCap
    df_opencap_side = align_columns(df_opencap_side, df_mocap, MARKER_MAPPING)
    df_opencap_front = align_columns(df_opencap_front, df_mocap, MARKER_MAPPING)
    #display fps
    print(f"-> OpenCap Side: {fps_opencap_side}Hz | OpenCap Front: {fps_opencap_front}Hz | MoCap: {fps_mocap}Hz")

    mapped_cols = set(df_opencap_side.columns)
    unmatched_cols = [col for col in df_mocap.columns if col not in mapped_cols and col not in ['Frame#', 'Time']]

    if unmatched_cols:
        unique_unmatched_markers = sorted(list(set([c.rsplit('_', 1)[0] for c in  unmatched_cols])))
        print(f"     {unique_unmatched_markers}")

    mocap_cols_to_keep = [col for col in df_mocap.columns if col in mapped_cols or col in ['Time']]
    df_mocap = df_mocap[mocap_cols_to_keep]

    #find overlapping time window
    start_time = max(df_opencap_side['Time'].min(), df_opencap_front['Time'].min(), df_mocap['Time'].min())
    end_time = min(df_opencap_side['Time'].max(), df_opencap_front['Time'].max(), df_mocap['Time'].max())

    #filter MoCap to only include overlapping window
    df_mocap_clipped = df_mocap[(df_mocap['Time'] >= start_time) & (df_mocap['Time'] <= end_time)].copy()
    target_time_vector = df_mocap_clipped['Time'].values

    def resample_dataframe(df_source, target_times):

        df_resampled = pd.DataFrame({'Time': target_times})
        cols = [col for col in df_source.columns if col not in ['Frame#', 'Time'] and not col.startswith('Unnamed')]

        for col in cols:
            if pd.api.types.is_numeric_dtype(df_source[col]):
                spline = CubicSpline(df_source['Time'], df_source[col])
                df_resampled[col] = spline(target_times)
        return df_resampled

    print("Resampling OpenCap Side data using Cubic Splines...")
    df_opencap_side_resampled = resample_dataframe(df_opencap_side, target_time_vector)

    print("Resampling OpenCap Front data using Cubic Splines...")
    df_opencap_front_resampled = resample_dataframe(df_opencap_front, target_time_vector)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mocap_out = output_path.replace('.csv', '_mocap_cleaned.csv')
    opencap_side_out = output_path.replace('.csv', '_opencap_side_resampled.csv')
    opencap_front_out = output_path.replace('.csv', '_opencap_front_resampled.csv')

    df_mocap_clipped.to_csv(mocap_out, index = False)
    df_opencap_side_resampled.to_csv(opencap_side_out, index=False)
    df_opencap_front_resampled.to_csv(opencap_front_out, index=False)


if __name__ == "__main__":
    
    opencapSide_folder = "raw_data/Opencap_Side/MarkerData/"
    opencapFront_folder = "raw_data/Opencap_Front/MarkerData/"
    mocap_folder = "raw_data/MoCap_Data/MarkerData/"
    output_folder = "processed_data/"

    search_pattern = os.path.join(opencapSide_folder, "*.trc")
    side_files = glob.glob(search_pattern)

    print(f"Found {len(side_files)} trials to process. Starting batch processing...\n")

    for side_path in side_files:
        filename = os.path.basename(side_path)
        base_name = filename.replace(".trc", "")

        front_path = os.path.join(opencapFront_folder, filename)
        mocap_path = os.path.join(mocap_folder, filename)

        if not os.path.exists(front_path):
            print(f"[SKIP] Missing front camera file for {base_name}. Looked in: {opencapFront_folder}")
            continue
        if not os.path.exists(mocap_path):
            print(f"[SKIP] Missing MoCap file for {base_name}. Looked in: {mocap_folder}")
            continue

        output_path = os.path.join(output_folder, f"aligned_{base_name}.csv")
        print(f"--- Processing Trial: {base_name} ---")
        process_and_align_trajectories(side_path, front_path, mocap_path, output_path)

    print("\n Batch processing completely finished!")

