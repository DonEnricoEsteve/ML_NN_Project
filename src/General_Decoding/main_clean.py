# Import necessary modules
from utils import *
from n01_data_preprocessing import *
from n02_general_decoding import *
from n03_plot_descriptive_and_results import *
from n05_RSA import *

# ============================================== #
# Part 1: Data preparation and preprocessing
# ============================================== #

# Convert data of class Epochs in MNE to a 3D numpy array. Do this for each time window (n=6) of interest
for npy, start, end, ppc_name in zip(npys, component_start_times, component_end_times, ppc_names):
    convert_epochsFIF_to_npy(fif_input_directory=fif_directory, npy_output_directory=npy, tmin=start, tmax=end)
    
    # Preprocess the numpy arrays of the 18 original conditions by pseudo-trial calculation
    # Derive new conditions (n=20) from the original 18 conditions
    for grp, ppc in zip(grps, ppc_folders):

        # Average over time and apply pseudo-trial calculation for the 2 grps
        # Suffix is an artifact of the previous pipeline
        preprocess_helper(input_dir=npy, output_dir=f"{ppc}/{ppc_name}", process_func=calculate_pseudo_trials, n_groups=grp, suffix=sf)
        
        # Define new class labels for the 2 ppcs
        derive_class_labels(npy_IO_directory=f"{ppc}/{ppc_name}", suffix=sf)


# ============================================== #
# Part 2: General decoding
# ============================================== #

# Define a wrapping function for parallelized across-subject decoding
def run_decoding(ppc_subfolder, fname, expl_var_perc):

    # Run binary classification (n=25 tasks)
    perform_general_decoding(base_dir=ppc_subfolder, tasks=binary_tasks, classification_type='binary', output_filename=fname,
                            variance_threshold=expl_var_perc, shuffle_labels=False)

    # Run multi-class classification (n=10 tasks)
    perform_general_decoding(base_dir=ppc_subfolder, tasks=multiclass_tasks, classification_type='multi', output_filename=fname,
                            variance_threshold=expl_var_perc, shuffle_labels=False)

# Parallelize general decoding over the sample-feature size combinations
Parallel(n_jobs=-1)(delayed(run_decoding)(ppc_subfolder, fname, expl_var_perc) 
                    for ppc_subfolder, fname, expl_var_perc in zip(ppc_subfolders, fnames, expl_var_percs))
    
# Evaluate the results of across-subject decoding using one-tailed Wilcoxon signed-rank test
for fname in fnames:

    # Load the resulting lists of within-subject accuracies for each task (n=35 total)
    list_binary_accuracies = np.load(f"{results_dir}/{fname}_accuracies_binary.npy")
    list_multi_accuracies = np.load(f"{results_dir}/{fname}_accuracies_multi.npy")

    # Perform statistical analysis for each task (Wilcoxon signed-rank test)
    evaluate_decoding(list_binary_accuracies=list_binary_accuracies, list_multi_accuracies=list_multi_accuracies,
                      binary_tasks=binary_tasks, multiclass_tasks=multiclass_tasks, multiclass_chance_levels=multi_chance)


# ============================================================ #
# Part 3: Plot summary statistics and general decoding results
# ============================================================ #

    # Figure 1 is a framework. This code is continuous with the for loop above
    # Plot Figure 2. Distribution of the difference between within-subject decoding accuracy and chance level for each task
    plot_distribution_diff(list_binary_accuracies=list_binary_accuracies, list_multi_accuracies=list_multi_accuracies,
                           binary_tasks=binary_tasks, multiclass_tasks=multiclass_tasks, multiclass_chance=multi_chance)

# Plot Figure 3. Across-subject decoding accuracies across windows and sample-feature sizes for decoding food and non-food 
plot_accuracy_trend(list_binary_acc=binary_files_for_Fig3)

# Plot Figure S1. Group-level (across 42 subjects) descriptive summary of MEG evoked responses over time by original category (n=3)
# Convert data of class Epochs in MNE to a 3D numpy array across the entire trial window. Used for plotting supplementary figures
convert_epochsFIF_to_npy(fif_input_directory=fif_directory, npy_output_directory=f"{wd}/ALL_orig", tmin=-0.300, tmax=0.799)

# Actually plot Figure S1.
for labels, conditions, txt, title in condition_groups:
    plot_summary_signals(final_rms=all_final_rms.item(), conditions=conditions, labels=labels, txt=txt, title=title)

# Plot Figure S2. Group-level (across 42 subjects) descriptive summary for MEG evoked responses by condition and window (n=5)
# Calculate summary statistics: root-mean square (RMS) and standard error of the mean across trials and channels
for rms_name, sem_name, time in zip(rms_names, sem_names, times):
    calculate_summary_statistics(input_dir=npys, rms_filename=rms_name, sem_filename=sem_name, time_truncate_ms=time)

# Actually plot Figure S2.
for (figtxt, window_label, color), rms_name in zip(components_labels, rms_names):
    # Load the RMS data for each time window
    rms_window = np.load(f"{SS_dir}/{rms_name}", allow_pickle=True)

    # Call the plotting function with the respective parameters for each component
    plot_summary_boxplots(final_rms=rms_window.item(), txt=figtxt, title=window_label, color=color)


# ============================================================ #
# Part 5: Representational Similarity Analysis (RSA)
# ============================================================ #

# Define a wrapping function for parallelized RSA
def process_ppc_save(rsa_ppc, save):
    # Concatenate all data in a single dictionary
    RSA_data_dict = populate_data_dict(npy_directory=rsa_ppc, cond_list=pca_orig_cond_list)

    # Perform RSA
    RSA_results = compute_pairwise_rsa(data_dict=RSA_data_dict, conditions=pca_orig_cond_list, 
                                       variance_threshold=0.90, save_filename=f"{save}_90_RSA")
    return RSA_results

# Parallelize RSA over the sample-feature size combinations and time windows
Parallel(n_jobs=-1)(delayed(process_ppc_save)(rsa_ppc, save) for rsa_ppc, save in zip(rsa_ppcs, component_labels))

# Optional: Parallelize RSA over the pairwise binary tasks (n=153) for a single time window and sample-feature size combination
# This was made as the baseline window was being left out in the parallelization process above
# RSA_data_dict = populate_data_dict(npy_directory=f"{wd}/10PT/BL_Evoked_PPC", cond_list=pca_orig_cond_list)
# RSA_results = compute_pairwise_rsa_parallel(data_dict=RSA_data_dict, conditions=pca_orig_cond_list, variance_threshold = 0.90, save_filename=f"BL_90_RSA")

# Plot representation dissimilarity matrix (RDM, Figure S3) and its corresponding multi-dimensional scaling (MDS) plot (Figure 6)
M100_results = np.load("/Volumes/Seagate/ML_Project/MARCH_ANALYSIS/RSA_Results_03_27/PS_90_RSA_within.npy", allow_pickle=True) # Change for window of interest
BL_results = np.load("/Volumes/Seagate/ML_Project/MARCH_ANALYSIS/RSA_Results_03_27/BL_90_RSA_within.npy", allow_pickle=True)

plot_confusion_matrix_with_stats(auc_arrays=M100_results.item(), baseline_arrays=BL_results.item(),
                                 conditions=pca_orig_cond_list, condition_labels=cond_short_labels, 
                                 title1=f"Representational Dissimilarity Matrix during the Entire Post-Stimulus",
                                 title2=f"Multidimensional Scaling Plot of the RDM during the Entire Post-Stimulus")
