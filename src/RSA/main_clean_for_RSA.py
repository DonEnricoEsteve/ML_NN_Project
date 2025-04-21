# Import necessary modules
from utils import *
from n05_RSA import *

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
