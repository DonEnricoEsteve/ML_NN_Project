import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from config import *  # Import configuration variables like paths, resampling rate, alphas
import mne  # MNE for EEG/MEG data analysis

def plot_sliding(contrast: str):
    """
    Function that plots decoding performance across time using the sliding estimator.
    It includes AUC scores, chance level, standard error, and significant time points based on p-values.

    Parameters:
    * contrast (str): The name of the contrast being decoded (e.g., "food-neutral").

    Returns: None. Saves the decoding figure with significance and prints the max decoding score.
    """
    # Load and resample the epochs for subject 003
    epochs_decim = mne.read_epochs(f"{epochs_path}/sub003_epo.fif").resample(resampling_rate, verbose=False)
    time_points = epochs_decim.times  # Extract time points from the epochs
    del epochs_decim  # Free memory after getting time points

    # Load decoding scores, p-values, and SEM for sliding estimator
    sliding_scores_all = np.load(f"{scores_path}/sliding_estimator_{contrast}_scores.npy")
    pvalue_arr = np.load(f"{stats_path}/sliding_estimator_{contrast}_pvalues.npy")
    sem_arr = np.load(f"{stats_path}/sliding_estimator_{contrast}_sem.npy")

    # Initialize dictionaries to store significant indices and plotting values
    significant_idx = {}
    times_sig = {}
    sig_val = {}

    # Predefined y-axis values for plotting significance points and their corresponding colors
    plot_sig_row = [0.47, 0.46, 0.45] 
    color_list = ["midnightblue", "gold", "plum"] 

    # Loop over different alpha levels to find significant time points
    for i in range(len(alphas)):
        significant_idx[i] = (pvalue_arr < alphas[i]).reshape(-1,)  # Boolean mask of significant p-values
        times_sig[i] = time_points[significant_idx[i]]  # Extract corresponding time points
        sig_val[i] = np.array([plot_sig_row[i]]*len(times_sig[i]))  # Set y-values for scatter plot

    _, ax = plt.subplots()  # Create a subplot

    # Scatter plot of significant time points for each alpha threshold
    scatter_handles = []
    for i in range(len(alphas)):
        scatter_handle = ax.scatter(times_sig[i], sig_val[i], c=color_list[i], s=20, label=f"pvalue < {alphas[i]}")
        scatter_handles.append(scatter_handle)
    scatter_legend = ax.legend(handles=scatter_handles, loc="upper left")  # Add legend for significance

    # Compute mean scores and plot them
    mean_sliding_scores = np.mean(sliding_scores_all, axis=0)
    scores_handle = ax.plot(time_points, mean_sliding_scores, label="score")[0]
    chance_handle = ax.axhline(0.5, color="r", linestyle="--", label="chance")  # Plot chance line
    plt.legend(handles=[scores_handle, chance_handle], loc="upper right")  # Add performance legend
    ax.add_artist(scatter_legend)  # Add scatter legend separately

    # Plot the standard error of the mean around the average scores
    plt.fill_between(time_points, mean_sliding_scores + np.array(sem_arr), mean_sliding_scores - np.array(sem_arr), color="lightsteelblue")

    # Add labels and title to the plot
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("AUC")  # Area Under the Curve
    ax.set_title(f"Decodability of {contrast} contrast across time") 
    ax.axvline(0.0, color="k", linestyle="-")  # Mark stimulus onset

    # Print the maximum decoding performance and its corresponding time
    print(f"Max decoding - [time (sec): {round(time_points[np.argmax(mean_sliding_scores)]*1000)}, AUC value: {np.round(np.max(mean_sliding_scores), 2)}]")

    # Save the figure
    plt.savefig(fname=f"{plots_path}/sliding_scores")


def plot_generalizing(contrast: str): 
    """
    Function that visualizes generalization across time using the generalizing estimator.
    It includes a performance matrix (AUC) and a significance matrix (p-values < 0.05).

    Parameters:
    * contrast (str): The name of the contrast being decoded ("food-neutral").

    Returns: None. Saves two figures: the decoding matrix and the p-value significance matrix.
    """
    # Load generalization scores and p-values
    generalizing_scores_all = np.load(f"{scores_path}/generalizing_estimator_{contrast}_scores.npy")
    pvalue_arr = np.load(f"{stats_path}/generalizing_estimator_{contrast}_pvalues.npy")
    
    # Load and resample epochs to get time points
    epochs_decim = mne.read_epochs(f"{epochs_path}/sub003_epo.fif").resample(resampling_rate, verbose=False)
    time_points = epochs_decim.times
    del epochs_decim

    # Compute mean decoding scores across subjects
    scores_mean = np.mean(generalizing_scores_all, axis=0)

    # Create a diverging color map centered around chance level (0.5)
    norm = mcolors.TwoSlopeNorm(vmin=np.min(scores_mean), vcenter=0.5, vmax=np.max(scores_mean))

    # Plot the generalization matrix
    fig, ax_scores = plt.subplots(layout="constrained")
    im_scores = ax_scores.matshow(
        scores_mean,
        norm=norm,
        cmap="RdBu_r",
        origin="lower",
        extent=time_points[[0, -1, 0, -1]],  # Set time axes
    )

    # Plot lines indicating stimulus onset
    ax_scores.axhline(0.0, color="k")
    ax_scores.axvline(0.0, color="k")
    ax_scores.xaxis.set_ticks_position("bottom")  # Move x-axis to bottom
    ax_scores.set_xlabel('Testing Time (s)')
    ax_scores.set_ylabel('Training Time (s)')
    ax_scores.set_title(f"Generalization across time for {contrast} contrast", fontweight="bold")
    fig.colorbar(im_scores, ax=ax_scores, label="Performance (ROC AUC)")  # Add color bar

    # Save the generalization matrix figure
    plt.savefig(fname=f"{plots_path}/generalizing_scores")

    # Plot the p-value matrix
    fig, ax_p = plt.subplots(layout="constrained")
    ax_p.matshow(
        pvalue_arr < 0.05,  # Boolean mask for significance
        vmin=0,
        vmax=1,
        cmap="RdBu_r",
        origin="lower",
        extent=time_points[[0, -1, 0, -1]],
    )

    # Add legend showing p<0.05 significance
    cmap = plt.get_cmap("RdBu_r")
    red_patch = mpatches.Patch(color=mcolors.to_hex(cmap(1.0)), label='p<0.05')
    ax_p.legend(handles=[red_patch], loc="upper right")

    # Stimulus onset markers
    ax_p.axhline(0.0, color="k")
    ax_p.axvline(0.0, color="k")
    ax_p.xaxis.set_ticks_position("bottom")
    ax_p.set_xlabel('Testing Time (s)')
    ax_p.set_ylabel('Training Time (s)')
    ax_p.set_title(f"Generalization across time for {contrast} contrast", fontweight="bold")

    # Save the p-value matrix figure
    plt.savefig(fname=f"{plots_path}/generalizing_pvalues")

    # Show all plots
    plt.show()