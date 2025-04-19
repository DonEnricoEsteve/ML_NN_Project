import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def compute_haufe_activation(weights, cov_X):
    """
    Perform Haufe transformation of classifier weights.
    """
    norm_factor = np.dot(weights, cov_X @ weights.T)[0, 0]
    activation = cov_X @ weights.T / norm_factor
    return activation.flatten()

def visualize_haufe_projection(weight_output_dir, combined_conditions_dir, haufe_map_dir,
                                bti_hdr_file, bti_config_file, head_shape_file,
                                haufe_weight_dir, vmin=-1, vmax=1):
    """
    Apply Haufe transformation on classifier weights and visualize activation maps.
    Save both activation arrays and topomap images.
    """
    os.makedirs(haufe_map_dir, exist_ok=True)
    os.makedirs(haufe_weight_dir, exist_ok=True)

    raw = mne.io.read_raw_bti(bti_hdr_file, bti_config_file, head_shape_file, preload=True)
    info = raw.info

    files = sorted(f for f in os.listdir(weight_output_dir) if f.endswith("_weights.npy"))
    for fname in files:
        subject_id, time_window = fname.replace("_weights.npy", "").split("_", 1)

        food_path = os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_food_combined.npy")
        nonfood_path = os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_nonfood_combined.npy")
        weight_path = os.path.join(weight_output_dir, fname)

        if not os.path.exists(food_path) or not os.path.exists(nonfood_path):
            logger.info(f"Skipping {subject_id}-{time_window}: missing condition files")
            continue

        food_data = np.load(food_path)
        nonfood_data = np.load(nonfood_path)
        pseudo_data = np.vstack([food_data, nonfood_data])
        n_channels = pseudo_data.shape[1]

        X_flat = pseudo_data.reshape(pseudo_data.shape[0], -1)
        cov_X = np.cov(X_flat, rowvar=False)

        weights = np.load(weight_path)
        activation = compute_haufe_activation(weights, cov_X)
        activation_2d = activation.reshape(n_channels, -1)
        avg_activation = activation_2d.mean(axis=1)

        meg_picks = mne.pick_types(info, meg=True, exclude='bads')
        info_meg = mne.pick_info(info, meg_picks[:n_channels])

        haufe_weight_path = os.path.join(haufe_weight_dir, f"{subject_id}_{time_window}_haufe.npy")
        np.save(haufe_weight_path, avg_activation)

        fig, ax = plt.subplots()
        mne.viz.plot_topomap(
            avg_activation,
            pos=info_meg,
            axes=ax,
            show=False,
            sensors=True,
            outlines='head',
            contours=4,
            res=256,
            extrapolate='local',
            vlim=(vmin, vmax)
        )
        ax.set_title(f"{subject_id}-{time_window}\nRed=Food, Blue=Non-Food")
        save_path = os.path.join(haufe_map_dir, f"{subject_id}_{time_window}_haufe.png")
        plt.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved Haufe topomap: {save_path}")
