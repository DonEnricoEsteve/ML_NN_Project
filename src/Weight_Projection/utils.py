import os

# === BASE DIRECTORY ===
base_dir = "/Users/folasewaabdulsalam/MEG_DECODING_ML/ML_NN_Project"

# === PATH CONFIG ===
evoked_data_dir = os.path.join(base_dir, "Evoked_fif")
time_window_evoked_dir = os.path.join(base_dir, "time_window_evoked")
condition_split_dir = os.path.join(base_dir, "condition_split")
pseudo_trial_dir = os.path.join(base_dir, "pseudo_trial")
combined_conditions_dir = os.path.join(base_dir, "condition_combined")
weight_output_dir = os.path.join(base_dir, "classifier_weights")
haufe_weight_dir = os.path.join(base_dir, "haufe_weights")
haufe_map_dir = os.path.join(base_dir, "haufe_maps")
haufe_group_map_dir = os.path.join(base_dir, "group_topomaps")
output_plot_dir = os.path.join(base_dir, "check-baseline")
sensor_summary_dir = os.path.join(base_dir, "sensor_summary")
annotated_map_dir = os.path.join(base_dir, "M100_Map")

# === BTI FILES ===
bti_hdr_file = os.path.join(base_dir, "c,rfhp0.1Hz")
bti_config_file = os.path.join(base_dir, "config")
head_shape_file = os.path.join(base_dir, "hs_file")

# === TIME WINDOWS ===
time_windows = {
    "Baseline": (-0.300, 0),
    "M100": (0.118, 0.155),
    "M200": (0.171, 0.217),
    "M300": (0.239, 0.332),
    "LPP": (0.350, 0.800),
    "Post-Stimulus": (0, 0.800)
}

# === TOP SENSOR LIST ===
M100_top_sensors = [
    "MEG 168", "MEG 055", "MEG 127", "MEG 154", "MEG 188",
    "MEG 056", "MEG 134", "MEG 163", "MEG 156", "MEG 197"
]

