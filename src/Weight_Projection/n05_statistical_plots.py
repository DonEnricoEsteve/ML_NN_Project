import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import mne
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def plot_top_channels_from_haufe(
    haufe_weight_dir,
    output_plot_dir,
    bti_hdr_file,
    bti_config_file,
    head_shape_file,
    top_n=10
):
    os.makedirs(output_plot_dir, exist_ok=True)

    raw = mne.io.read_raw_bti(bti_hdr_file, bti_config_file, head_shape_file, preload=True)
    meg_picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    ch_names_full = raw.info['ch_names']

    files = sorted(f for f in os.listdir(haufe_weight_dir) if f.endswith("_haufe.npy"))
    time_windows = sorted(set(f.split("_")[1] for f in files))
    all_windows = ["Baseline", "Post-Stimulus", "M100", "M200", "M300","LPP" ]

    summary = []
    for window in time_windows:
        matched_files = [f for f in files if f"_{window}_haufe.npy" in f]
        if not matched_files:
            continue

        activations = [np.load(os.path.join(haufe_weight_dir, f)) for f in matched_files]
        activations = np.stack(activations)
        avg_activation = np.mean(activations, axis=0)
        n_channels = avg_activation.shape[0]
        ch_names = [ch_names_full[i] for i in meg_picks[:n_channels]]

        abs_activation = np.abs(avg_activation)
        top_idx = np.argsort(abs_activation)[-top_n:][::-1]
        top_names = [ch_names[i] for i in top_idx]
        top_values = avg_activation[top_idx]

        for name, val in zip(top_names, top_values):
            summary.append({'time_window': window, 'channel': name, 'activation': val})

        plt.figure(figsize=(8, 4))
        plt.bar(top_names, top_values, color='red')
        plt.xticks(rotation=45)
        plt.title(f"Group Avg Top {top_n} Channels - {window}")
        plt.ylabel("Mean Haufe Activation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, f"group_top_{top_n}_{window}.png"))
        plt.close()

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_plot_dir, f"top_{top_n}_channels_summary.csv"), index=False)

    selected_channels = ["MEG 055", "MEG 168"]
    subject_ids = sorted(set(f.split("_")[0] for f in files))
    channel_activations = []
    for subj in subject_ids:
        for window in all_windows:
            fpath = os.path.join(haufe_weight_dir, f"{subj}_{window}_haufe.npy")
            if not os.path.exists(fpath):
                continue
            data = np.load(fpath)
            for ch_name in selected_channels:
                try:
                    ch_idx = ch_names_full.index(ch_name)
                    activation = data[ch_idx]
                    channel_activations.append({
                        "subject": subj,
                        "time_window": window,
                        "channel": ch_name,
                        "activation": activation
                    })
                except ValueError:
                    continue

    df_subjects = pd.DataFrame(channel_activations)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    labels = {
        "MEG 055": "MEG 055 (Right Temporal)",
        "MEG 168": "MEG 168 (Occipital)"
    }

    for idx, ch in enumerate(selected_channels):
        df_ch = df_subjects[df_subjects['channel'] == ch]
        grouped = df_ch.groupby("time_window")["activation"]
        mean_acts = grouped.mean().reindex(all_windows)
        sem_acts = grouped.sem().reindex(all_windows)
        color = 'steelblue' if idx == 0 else 'indianred'
        bars = axes[idx].bar(mean_acts.index, np.abs(mean_acts.values), yerr=sem_acts.values,
                             capsize=5, color=color, edgecolor='black')

        axes[idx].set_title(labels.get(ch, ch))
        axes[idx].set_xlabel("Time Window")
        if idx == 0:
            axes[idx].set_ylabel("Mean Haufe Activation")

        #Significance lines
        y_max = (np.abs(mean_acts) + sem_acts).max()
        h = 0.05 * y_max
        line_height = y_max + h
        bar_centers = {label: bar.get_x() + bar.get_width() / 2 for label, bar in zip(all_windows, bars)}

        # comparing M100 to each other time window
        comparisons = [("M100", w) for w in all_windows if w != "M100"]

        for i, (w1, w2) in enumerate(comparisons):
            group1 = df_ch[df_ch["time_window"] == w1].sort_values("subject")["activation"].values
            group2 = df_ch[df_ch["time_window"] == w2].sort_values("subject")["activation"].values
            if len(group1) != len(group2) or len(group1) == 0:
                continue
            stat, pval = ttest_rel(group1, group2)
            if pval < 0.05:
                if pval < 0.001:
                    p_text = '***'
                elif pval < 0.01:
                    p_text = '**'
                else:
                    p_text = '*'
                x1, x2 = bar_centers[w1], bar_centers[w2]
                y = line_height + i * h
                axes[idx].plot([x1, x1, x2, x2], [y, y + h / 4, y + h / 4, y], lw=1.2, c='k')
                axes[idx].text((x1 + x2) / 2, y + h / 2, p_text, ha='center', va='bottom', fontsize=13)

    for ax in axes:
        ax.set_ylim([0, line_height + len(comparisons) * h + 0.2])

    plt.tight_layout()
    save_path = os.path.join(output_plot_dir, "sensor_activation_pairwise_M100_vs_others.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved statistical bar plot with significance annotations to: {save_path}")
