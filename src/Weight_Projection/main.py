import os
import mne
import logging


from haufe_pipeline_preprocessing import load_and_crop_time_windows, split_into_conditions, generate_pseudotrials, combine_conditions
from haufe_pipeline_train_classifier import train_classifier
from haufe_transform import visualize_haufe_projection
from haufe_pipeline_statistical_plots import plot_top_channels_from_haufe
from haufe_pipeline_topomaps import compute_group_haufe_topomaps, summarize_top_sensors_across_subjects,plot_grand_average_topomap_with_top_sensors

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")

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

M100_top_sensors = ["MEG 168", "MEG 055", "MEG 127", "MEG 154", "MEG 188",
                    "MEG 056", "MEG 134", "MEG 163", "MEG 156", "MEG 197"]

# === EXECUTE PIPELINE ===
load_and_crop_time_windows(evoked_data_dir, time_window_evoked_dir, time_windows)
split_into_conditions(time_window_evoked_dir, condition_split_dir)
generate_pseudotrials(condition_split_dir, pseudo_trial_dir)
combine_conditions(pseudo_trial_dir, combined_conditions_dir)
train_classifier(combined_conditions_dir, weight_output_dir, time_windows)
visualize_haufe_projection(weight_output_dir, combined_conditions_dir, haufe_map_dir,
                           bti_hdr_file, bti_config_file, head_shape_file, haufe_weight_dir)
compute_group_haufe_topomaps(haufe_weight_dir, haufe_group_map_dir,
                             bti_hdr_file, bti_config_file, head_shape_file)
plot_top_channels_from_haufe(haufe_weight_dir, output_plot_dir,
                             bti_hdr_file, bti_config_file, head_shape_file, top_n=10)
summarize_top_sensors_across_subjects(haufe_weight_dir, sensor_summary_dir,
                                      bti_hdr_file, bti_config_file, head_shape_file, top_n=10)
plot_grand_average_topomap_with_top_sensors(
    haufe_weight_dir, M100_top_sensors, bti_hdr_file, bti_config_file,
    head_shape_file, annotated_map_dir, vmin=-1, vmax=1
)
