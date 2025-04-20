import os
import numpy as np
import mne
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_and_crop_time_windows(evoked_data_dir, time_window_evoked_dir, time_windows):
    """
    Crop evoked epochs into specified time windows and save them.
    """
    os.makedirs(time_window_evoked_dir, exist_ok=True)
    for fname in sorted(os.listdir(evoked_data_dir)):
        if fname.endswith("_epo.fif"):
            subject_id = fname.split("_epo.fif")[0]
            filepath = os.path.join(evoked_data_dir, fname)
            logger.info(f"Processing {subject_id}...")
            epochs = mne.read_epochs(filepath, preload=True)
            for win_name, (tmin, tmax) in time_windows.items():
                epoch_copy = epochs.copy().crop(tmin=tmin, tmax=tmax)
                out_name = f"{subject_id}_{win_name}.fif"
                save_path = os.path.join(time_window_evoked_dir, out_name)
                epoch_copy.save(save_path, overwrite=True)
                logger.info(f"Saved: {out_name}")

def split_into_conditions(time_window_evoked_dir, condition_split_dir):
    """
    Split each time-windowed epoch file into its 18 condition-specific epochs.
    """
    os.makedirs(condition_split_dir, exist_ok=True)
    condition_labels = [
        "food/short/rep1", "food/medium/rep1", "food/long/rep1",
        "food/short/rep2", "food/medium/rep2", "food/long/rep2",
        "positive/short/rep1", "positive/medium/rep1", "positive/long/rep1",
        "positive/short/rep2", "positive/medium/rep2", "positive/long/rep2",
        "neutral/short/rep1", "neutral/medium/rep1", "neutral/long/rep1",
        "neutral/short/rep2", "neutral/medium/rep2", "neutral/long/rep2"]
    for file in os.listdir(time_window_evoked_dir):
        if file.endswith(".fif"):
            filepath = os.path.join(time_window_evoked_dir, file)
            epochs = mne.read_epochs(filepath, preload=True)
            subject_tag = file.replace(".fif", "")
            for condition in condition_labels:
                if condition in epochs.event_id:
                    try:
                        condition_epoch = epochs[condition]
                        out_name = f"{subject_tag}_{condition.replace('/', '_')}.fif"
                        save_path = os.path.join(condition_split_dir, out_name)
                        condition_epoch.save(save_path, overwrite=True)
                        logger.info(f"Saved: {out_name}")
                    except Exception as e:
                        logger.info(f"Could not save {condition} for {file}: {e}")
                else:
                    logger.info(f"Condition {condition} not in {file}. Skipping ...")

def generate_pseudotrials(condition_split_dir, pseudo_trial_dir, n_groups=10):
    """
    Generate pseudotrials from condition-split epochs and average across time.
    """
    os.makedirs(pseudo_trial_dir, exist_ok=True)
    for file in sorted(os.listdir(condition_split_dir)):
        if not file.endswith('.fif'):
            continue
        file_path = os.path.join(condition_split_dir, file)
        epochs = mne.read_epochs(file_path, preload=True)
        data = epochs.get_data()
        n_trials, n_channels, n_times = data.shape
        if n_trials < n_groups:
            logger.info(f"Not enough trials in {file} to generate {n_groups} pseudotrials. Skipping ...")
            continue
        group_size = n_trials // n_groups
        groups = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]
        if n_trials % n_groups != 0:
            leftovers = np.arange(n_groups * group_size, n_trials)
            groups[-1] = np.concatenate([groups[-1], leftovers])
        pseudo_trials = []
        for group in groups:
            group_data = data[group]
            mean_over_trials = np.mean(group_data, axis=0)
            mean_over_time = np.mean(mean_over_trials, axis=1)
            pseudo_trials.append(mean_over_time)
        pseudo_trials = np.stack(pseudo_trials)
        base_name = os.path.splitext(file)[0]
        save_path = os.path.join(pseudo_trial_dir, f"{base_name}_pseudo.npy")
        np.save(save_path, pseudo_trials)
        logger.info(f"Saved: {base_name}")


def combine_conditions(pseudo_trial_dir, combined_conditions_dir):
    """
    Combine condition-specific pseudo-trials into binary categories: food vs nonfood.
    """
    os.makedirs(combined_conditions_dir, exist_ok=True)

    food_conditions = [
        "food_short_rep1", "food_short_rep2",
        "food_medium_rep1", "food_medium_rep2",
        "food_long_rep1", "food_long_rep2"
    ]

    nonfood_conditions = [
        "positive_short_rep1", "positive_short_rep2",
        "positive_medium_rep1", "positive_medium_rep2",
        "positive_long_rep1", "positive_long_rep2",
        "neutral_short_rep1", "neutral_short_rep2",
        "neutral_medium_rep1", "neutral_medium_rep2",
        "neutral_long_rep1", "neutral_long_rep2"
    ]

    subject_window_data = {}

    for filename in sorted(os.listdir(pseudo_trial_dir)):
        if filename.endswith("_pseudo.npy"):
            subject_id, time_window, condition = filename.replace("_pseudo.npy", "").split("_", 2)
            key = (subject_id, time_window)
            file_path = os.path.join(pseudo_trial_dir, filename)
            data = np.load(file_path)

            if key not in subject_window_data:
                subject_window_data[key] = {"food": [], "nonfood": []}

            if condition in food_conditions:
                subject_window_data[key]["food"].append(data)
            elif condition in nonfood_conditions:
                subject_window_data[key]["nonfood"].append(data)

    for (subject_id, time_window), categories in subject_window_data.items():
        if categories["food"]:
            food_combined = np.vstack(categories["food"])
            np.save(os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_food_combined.npy"), food_combined)
        if categories["nonfood"]:
            nonfood_combined = np.vstack(categories["nonfood"])
            np.save(os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_nonfood_combined.npy"), nonfood_combined)

    logger.info(f"Combined food and nonfood data saved to: {combined_conditions_dir}")
