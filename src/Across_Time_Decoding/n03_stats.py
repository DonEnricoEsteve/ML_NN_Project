from scipy.stats import wilcoxon, sem
import numpy as np

def stats_sliding(scores_all_subs: np.ndarray) -> tuple[list, list]:
    """
    Performs a Wilcoxon test on sliding_estimator AUC scores.

    Parameters:
        scores_all_subs: np.ndarray
            AUC scores of all subjects for each time point (n_subjects x n_time_points).

    Returns: 
        tuple:
            pvalue_list: list of p-values for each time point.
            sem_list: list of standard errors of the mean for each time point.
    """

    pvalue_list = []  # To store p-values for each time point
    sem_list = []     # To store SEM for each time point

    # Iterate over time points
    for i in range(scores_all_subs.shape[1]):
        subs_scores_at_time_point = scores_all_subs[:, i]  # AUC scores across subjects at current time point
        diff = subs_scores_at_time_point - 0.5  # Difference from chance level (0.5)
        
        sem_list.append(sem(subs_scores_at_time_point))  # Compute and store standard error of the mean
        
        # Perform one-sided Wilcoxon signed-rank test against 0.5 (chance level)
        _, pvalue = wilcoxon(diff, alternative='greater', method='auto', keepdims=True)
        pvalue_list.append(pvalue)  # Store the p-value
    
    return (pvalue_list, sem_list)

def stats_generalizing(scores_all_subs: np.ndarray) -> np.ndarray:
    """
    Performs a Wilcoxon test on time generalization estimator AUC scores.

    Parameters:
        scores_all_subs: np.ndarray
            AUC scores for each test-train time point (n_subjects x n_time_points x n_time_points).

    Returns:
        p_value_mat (np.ndarray): p-value matrix (n_time_points x n_time_points).
    """

    # Initialize matrix to hold p-values for each train-test time combination
    p_values_mat = np.zeros((220, 220))

    # Iterate over all time x time points
    for i in range(scores_all_subs.shape[1]):
        for j in range(scores_all_subs.shape[2]):
            subs_scores_point = scores_all_subs[:, i, j]  # AUC scores for subjects at train=i, test=j
            diff = subs_scores_point - 0.5  # Difference from chance
            
            # Perform one-sided Wilcoxon signed-rank test
            result = wilcoxon(diff, alternative="greater", method="auto", keepdims=True)
            p_values_mat[i, j] = result.pvalue  # Store the p-value

    return p_values_mat  # Return full p-value matrix
