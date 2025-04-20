# Contrasts and event id definition:
import numpy as np

orig_event_ids = {
    "food/short/rep1": 10, "food/medium/rep1": 12, "food/long/rep1": 14, 
    "food/short/rep2": 20, "food/medium/rep2": 22, "food/long/rep2": 24,
    "positive/short/rep1": 110, "positive/medium/rep1": 112, "positive/long/rep1": 114,
    "positive/short/rep2": 120, "positive/medium/rep2": 122, "positive/long/rep2": 124,
    "neutral/short/rep1": 210, "neutral/medium/rep1": 212, "neutral/long/rep1": 214,
    "neutral/short/rep2": 220, "neutral/medium/rep2": 222, "neutral/long/rep2": 224
}

F_events_code = np.array([10, 12, 14, 20, 22, 24])
N_events_code = np.array([110, 112, 114, 120, 122, 124, 210, 212, 214, 220, 222, 224])
F_N_events_code = np.append(F_events_code, N_events_code)
F_P_events_code = np.append(F_events_code, np.array([110, 112, 114, 120, 122, 124]))
F_T_events_code = np.append(F_events_code, np.array([210, 212, 214, 220, 222, 224]))

contrasts_events_code = {"food-nonfood": F_N_events_code, "food-positive": F_P_events_code, "food-neutral": F_T_events_code}

# Variable initialization:

time_points = 1119

trials_to_retain = 10 # Number of pseudo trials

n_channels = 246

resampling_rate = 200

alphas = [0.05, 0.01, 0.001]

# Paths:

# Set your own project path
project_path = "/Documents/ML_NN_Project"

scores_path = project_path + "/scores"

stats_path = project_path + "/stats"

epochs_path = project_path + "/Evoked_fif"

plots_path = project_path + "/plots"
