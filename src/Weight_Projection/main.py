import os
import mne
import logging


from haufe_pipeline_preprocessing import load_and_crop_time_windows, split_into_conditions, generate_pseudotrials, combine_conditions
from haufe_pipeline_train_classifier import train_classifier
from haufe_transform import visualize_haufe_projection
from haufe_pipeline_statistical_plots import plot_top_channels_from_haufe
from haufe_pipeline_topomaps import compute_group_haufe_topomaps, summarize_top_sensors_across_subjects,plot_grand_average_topomap_with_top_sensors
from utils import (base_dir, evoked_data_dir, time_window_evoked_dir, condition_split_dir, pseudo_trial_dir, combined_conditions_dir, weight_output_dir, haufe_weight_dir, haufe_map_dir, haufe_group_map_dir, output_plot_dir,sensor_summary_dir, annotated_map_dir, bti_hdr_file, bti_config_file,head_shape_file, time_windows, M100_top_sensors)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
mne.set_log_level("ERROR")


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
