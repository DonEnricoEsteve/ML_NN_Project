import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.channels.layout import _find_topomap_coords
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def compute_group_haufe_topomaps(haufe_weight_dir, haufe_group_map_dir,
                                  bti_hdr_file, bti_config_file, head_shape_file,
                                  vmin=-1, vmax=1, output_name="group_haufe_all_windows.png"):
    """
    Compute and visualize group-level average Haufe activation topomaps for each time window,
    and save them all in a single horizontally stacked image.
    """
    os.makedirs(haufe_group_map_dir, exist_ok=True)
    raw = mne.io.read_raw_bti(bti_hdr_file, bti_config_file, head_shape_file, preload=True)
    info = raw.info
    meg_picks = mne.pick_types(info, meg=True, exclude='bads')

    all_files = sorted(f for f in os.listdir(haufe_weight_dir) if f.endswith("_haufe.npy"))
    time_windows = sorted(set(f.split("_", 1)[1].replace("_haufe.npy", "") for f in all_files))

    fig, axes = plt.subplots(1, len(time_windows), figsize=(4 * len(time_windows), 4))
    if len(time_windows) == 1:
        axes = [axes]

    for i, time_window in enumerate(time_windows):
        activations = [
            np.load(os.path.join(haufe_weight_dir, f))
            for f in all_files if f"_{time_window}_haufe.npy" in f
        ]
        if not activations:
            print(f"No data for time window {time_window}")
            continue

        avg_activation = np.mean(activations, axis=0)
        n_channels = avg_activation.shape[0]
        info_meg = mne.pick_info(info, meg_picks[:n_channels])

        mne.viz.plot_topomap(
            avg_activation,
            pos=info_meg,
            axes=axes[i],
            show=False,
            sensors=True,
            outlines='head',
            contours=4,
            res=256,
            extrapolate='local',
            vlim=(vmin, vmax)
        )
        axes[i].set_title(f"{time_window}\nRed=Food, Blue=Non-Food", fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(haufe_group_map_dir, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved stacked group topomap image: {output_path}")

def summarize_top_sensors_across_subjects(haufe_weight_dir, output_dir,
                                          bti_hdr_file, bti_config_file, head_shape_file,
                                          top_n=10):
    """
    Summarize top-N most frequently activated sensors across all subjects and time windows.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = mne.io.read_raw_bti(bti_hdr_file, bti_config_file, head_shape_file, preload=True)
    info = raw.info
    meg_picks = mne.pick_types(info, meg=True, exclude='bads')
    ch_names = [info['ch_names'][i] for i in meg_picks]

    freq_counter = Counter()
    for fname in sorted(os.listdir(haufe_weight_dir)):
        if fname.endswith("_haufe.npy"):
            activation = np.load(os.path.join(haufe_weight_dir, fname))
            n_channels = activation.shape[0]
            selected_ch_names = ch_names[:n_channels]
            top_idx = np.argsort(np.abs(activation))[-top_n:]
            top_sensors = [selected_ch_names[i] for i in top_idx]
            freq_counter.update(top_sensors)

    top_df = pd.DataFrame(freq_counter.most_common(top_n), columns=["Sensor", "Frequency"])

    plt.figure(figsize=(16, 6))
    plt.bar(top_df["Sensor"], top_df["Frequency"], color="skyblue")
    plt.xlabel("Sensor Name")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} Sensors by Haufe Activation Frequency")
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"top_{top_n}_sensors_barplot.png")
    csv_path = os.path.join(output_dir, f"top_{top_n}_sensors.csv")
    plt.savefig(fig_path)
    plt.close()
    top_df.to_csv(csv_path, index=False)
    logger.info(f"Saved plot to {fig_path}")
    logger.info(f"Saved summary CSV to {csv_path}")

def plot_grand_average_topomap_with_top_sensors(haufe_weight_dir, top_sensors,
                                                 bti_hdr_file, bti_config_file,
                                                 head_shape_file, output_dir,
                                                 vmin=None, vmax=None):
    """
    Plot grand-average M100 topomap and annotate top sensor names with ranks.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_activations = []
    for fname in sorted(os.listdir(haufe_weight_dir)):
        if fname.endswith("_M100_haufe.npy"):
            data = np.load(os.path.join(haufe_weight_dir, fname))
            all_activations.append(data)

    if not all_activations:
        raise RuntimeError("No Haufe activations found.")

    grand_avg = np.mean(all_activations, axis=0)

    raw = mne.io.read_raw_bti(bti_hdr_file, bti_config_file, head_shape_file, preload=False)
    info = raw.info
    picks = mne.pick_types(info, meg=True, exclude='bads')
    ch_names = [info['ch_names'][i] for i in picks]
    n_channels = grand_avg.shape[0]
    picks = picks[:n_channels]
    ch_names = ch_names[:n_channels]
    coords = _find_topomap_coords(info, picks=np.array(picks))

    mask = np.array([name in top_sensors for name in ch_names])

    fig, ax = plt.subplots(figsize=(6, 6))
    mne.viz.plot_topomap(
        grand_avg,
        pos=coords,
        axes=ax,
        show=False,
        sensors=True,
        outlines='head',
        sphere=(0., 0., 0., 0.18),
        contours=0,
        extrapolate='local',
        vlim=(vmin, vmax) if (vmin is not None and vmax is not None) else None,
        cmap='RdBu_r',
        res=256,
        mask=mask,
        mask_params=dict(marker='o', markerfacecolor='black', markeredgecolor='white', linewidth=0, markersize=8)
    )

    for rank, sensor in enumerate(top_sensors, 1):
        if sensor in ch_names:
            i = ch_names.index(sensor)
            x, y = coords[i]
            ax.text(x, y, f"{rank}. {sensor}", color='black', fontsize=8, ha='center', va='center')

    ax.set_title("Haufe Topomap for M100 Window\n(Top 10 sensors highlighted)", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "grand_avg_M100_topomap_with_top10.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved topomap to: {save_path}")
