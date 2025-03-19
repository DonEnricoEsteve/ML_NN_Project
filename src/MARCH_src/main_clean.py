# Import necessary modules
from utils import *
from n01_epochs_to_arrays import *
from n03_general_decoding import *
# from n04_plot_descriptive_and_results import *
# from n07_RSA import *

# ============================================== #
# Part 1: Convert MNE epochs to Numpy arrays
# ============================================== #

# # Take the 3D data from each epochs of each original condition
# convert_epochsFIF_to_npy(fif_input_directory=fif_directory, npy_output_directory=npy_directory, 
#                          tmin=component_start_times[0], tmax=component_end_times[0]) 

# # Average over time and apply pseudo-trial calculation for the 18 original conditions
# preprocess_helper(input_dir=npy_directory, output_dir=PPC_directory, 
#                   process_func=calculate_pseudo_trials, 
#                   n_groups=10, suffix=SF)

# # Define new class labels from the original conditions
# derive_class_labels(npy_IO_directory=PPC_directory, suffix=SF)

# # ============================================== #
# # Part 3: General decoding
# # ============================================== #

fname = "BL_10_90"

# Run binary classification (n=19 tasks)
# Define output filename (recommended convention is window_binsize_groups_components)
perform_general_decoding(base_dir=PPC_directory, tasks=binary_tasks, 
                         classification_type='binary', output_filename=f"{fname}",
                         variance_threshold=0.90, shuffle_labels=False)

# Run multi-class classification (n=9 tasks)
# Define output filename (recommended convention is window_binsize_groups_components)
perform_general_decoding(base_dir=PPC_directory, tasks=multiclass_tasks, 
                         classification_type='multi', output_filename=f"{fname}",
                         variance_threshold=0.90, shuffle_labels=False)

# Load the resulting lists (from general decoding) of within-subject accuracies for each task (n=18)
list_binary_accuracies = np.load(f"{results_dir}/{fname}_accuracies_binary.npy")
list_multi_accuracies = np.load(f"{results_dir}/{fname}_accuracies_multi.npy")

# # Perform statistical analysis for each task (Wilcoxon signed-rank test)
evaluate_decoding(list_binary_accuracies=list_binary_accuracies, list_multi_accuracies = list_multi_accuracies,
                  binary_tasks=binary_tasks, multiclass_tasks=multiclass_tasks,
                  binary_chance_levels=binary_chance, multiclass_chance_levels=multi_chance)

# PPCs = [f"{wd}/5PT/BL_Evoked_PPC", f"{wd}/5PT/PS_Evoked_PPC", f"{wd}/5PT/M100_Evoked_PPC", f"{wd}/5PT/M200_Evoked_PPC", f"{wd}/5PT/M300_Evoked_PPC", f"{wd}/5PT/MLPP_Evoked_PPC",
#         f"{wd}/5PT/BL_Evoked_PPC", f"{wd}/5PT/PS_Evoked_PPC", f"{wd}/5PT/M100_Evoked_PPC", f"{wd}/5PT/M200_Evoked_PPC", f"{wd}/5PT/M300_Evoked_PPC", f"{wd}/5PT/MLPP_Evoked_PPC",
#         f"{wd}/10PT/BL_Evoked_PPC", f"{wd}/10PT/PS_Evoked_PPC", f"{wd}/10PT/M100_Evoked_PPC", f"{wd}/10PT/M200_Evoked_PPC", f"{wd}/10PT/M300_Evoked_PPC", f"{wd}/10PT/MLPP_Evoked_PPC",
#         f"{wd}/10PT/BL_Evoked_PPC", f"{wd}/10PT/PS_Evoked_PPC", f"{wd}/10PT/M100_Evoked_PPC", f"{wd}/10PT/M200_Evoked_PPC", f"{wd}/10PT/M300_Evoked_PPC", f"{wd}/10PT/MLPP_Evoked_PPC"]

# threshs = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80,
#            0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
#            0.80, 0.80, 0.80, 0.80, 0.80, 0.80,
#            0.90, 0.90, 0.90, 0.90, 0.90, 0.90]

# for PPC, fname, thresh in zip(PPCs, fnames, threshs):

#     # Run binary classification (n=19 tasks)
#     # Define output filename (recommended convention is window_binsize_groups_components)
#     perform_general_decoding(base_dir=PPC, tasks=binary_tasks, 
#                             classification_type='binary', output_filename=f"{fname}",
#                             variance_threshold=thresh, shuffle_labels=False)

#     # Run multi-class classification (n=9 tasks)
#     # Define output filename (recommended convention is window_binsize_groups_components)
#     perform_general_decoding(base_dir=PPC, tasks=multiclass_tasks, 
#                             classification_type='multi', output_filename=f"{fname}",
#                             variance_threshold=thresh, shuffle_labels=False)

# for fname in fnames:

#     # Load the resulting lists (from general decoding) of within-subject accuracies for each task (n=18)
#     list_binary_accuracies = np.load(f"{results_dir}/{fname}_accuracies_binary.npy")
#     list_multi_accuracies = np.load(f"{results_dir}/{fname}_accuracies_multi.npy")

#     # # Perform statistical analysis for each task (Wilcoxon signed-rank test)
#     evaluate_decoding(list_binary_accuracies=list_binary_accuracies, list_multi_accuracies=list_multi_accuracies,
#                     binary_tasks=binary_tasks, multiclass_tasks=multiclass_tasks,
#                     binary_chance_levels=binary_chance, multiclass_chance_levels=multi_chance)

# # ============================================================ #
# # Part 4: Plot summary statistics and general decoding results
# # ============================================================ #

# # # Plot Figure 1. Distribution of the difference between within-subject decoding accuracy (n=42) and chance level for each classification task
# plot_distribution_diff(list_binary_accuracies=list_binary_accuracies, list_multi_accuracies=list_multi_accuracies,
#                        binary_tasks=binary_tasks, multiclass_tasks=multiclass_tasks,
#                        binary_chance=binary_chance, multiclass_chance=multi_chance)

# # Plot Figure 2. Across-subject decoding accuracies across different time windows for binary tasks involving the food and non-food contrasts 
# list_binary_acc = [np.load(f'{results_dir}/{file}') for file in binary_files_for_Fig2]
# plot_accuracy_trend(list_binary_acc=list_binary_acc)

# # Calculate summary statistics: root-mean square (RMS) and standard error of the mean across trials and channels
# final_rms, final_sem = calculate_summary_statistics(input_dir=input_dir_for_FigS1, rms_filename='poststimulus_RMS', sem_filename='poststimulus_SEM')

# # Plot Figure S1. Group-level (across 42 subjects) descriptive summary for MEG evoked responses by condition group (n=3)
# for labels, txt, label in condition_groups:
#     plot_summary_signals(final_rms=final_rms, final_sem=final_sem, labels=labels, txt=txt, label=label)

# # Plot Figure S2. Group-level (across 42 subjects) descriptive summary for MEG evoked responses by condition and window (n=5)
# for start, end, txt, title, color in components_with_colors:
#     plot_summary_boxplots(final_rms=final_rms, start_time_ms=start, end_time_ms=end, txt=txt, title=title, color=color)

# # ============================================================ #
# # Part 7: Representational Similarity Analysis (RSA)
# # ============================================================ #

# # Extract each condition array (.npy)
# RSA_convert_epochsFIF_to_npy(fif_input_directory=fif_directory, npy_output_directory=f"{RSA_wd}/conds",
#                              tmin=component_start_times[2], tmax=component_end_times[2])

# # Concatenate all data in a single dictionary
# RSA_data_dict = populate_data_dict(npy_directory=f"{RSA_wd}/M200_conds_PCA", cond_list=pca_orig_cond_list)

# # Perform RSA
# RSA_results = compute_pairwise_rsa(data_dict=RSA_data_dict, conditions=pca_orig_cond_list, save_filename='M200_RSA')

# # Plot representation dissimilarity matrix (RDM, Figure S3) and its corresponding multi-dimensional scaling (MDS) plot (Figure 5)
# plot_confusion_matrix_with_mds(auc_results=RSA_results.item(), conditions=pca_orig_cond_list, condition_labels=cond_short_labels,
#                                title1="Representational Dissimilarity Matrix (RDM) during the m200 window",
#                                title2="Multidimensional Scaling (MDS) Plot of the RDM during the m200 window")
