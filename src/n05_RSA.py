from utils import *

def populate_data_dict(npy_directory, cond_list):
    """
    Populate data_dict from .npy files stored in a directory for each subject.

    Parameters:
    - directory (str): The root directory where the condition data files are stored.
    - cond_list (list): List of condition names to search for.

    Returns:
    - data_dict (dict): Dictionary where keys are subject IDs and values are another dictionary with condition names as keys and np.array of data as values.
    """
    
    data_dict = {}

    # Find all subject folders starting with 'sub' (assuming folder names are like sub01, sub02, etc.)
    subject_folders = sorted(glob.glob(os.path.join(npy_directory, 'sub*')))
    
    # Iterate over each subject folder
    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)  # Assuming the folder name is the subject ID
        
        # Initialize a dictionary for the current subject
        subject_data = {}
        
        # Iterate over the condition list
        for cond in cond_list:
            # Search for the .npy file corresponding to the condition for the current subject
            file_pattern = os.path.join(subject_folder, f"{cond}.npy")
            condition_files = glob.glob(file_pattern)
            
            if condition_files:
                # Load the data if the file exists
                subject_data[cond] = np.load(condition_files[0])
            else:
                print(f"Warning: No file found for subject {subject_id} and condition {cond}")
        
        # Add the subject data to the main dictionary
        if subject_data:
            data_dict[subject_id] = subject_data
    
    return data_dict


def compute_pairwise_rsa(data_dict, conditions, variance_threshold, save_filename):
    """
    Perform RSA (pairwise binary classification) for each condition pair, and compute ROC-AUC scores.

    Parameters:
    - data_dict (dict): Dictionary with subjects as keys and their corresponding data as values.
    - conditions (list): List of conditions to compare.
    - save_filename (str): Name of file containing auc_scores

    Returns:
    - auc_scores (dict): Dictionary with condition pairs as keys and their corresponding AUC scores as values.
    """

    auc_scores = {}
    auc_scores_for_all_pairs = {}  # Dictionary to store auc_scores_for_pair for each pair
    logo = LeaveOneGroupOut()

    # Loop through all pairs of conditions
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i >= j:  # Avoid repeating pairs (cond1 vs cond2 and cond2 vs cond1)
                continue

            print(f"Processing pair: {cond1} vs {cond2}")

            # Initialize variables for stacking data and labels
            X = []
            y = []
            groups = []

            # Iterate over subjects and stack data for the two conditions
            for subject, subject_data in data_dict.items():
                # Check if both conditions exist for the subject
                if cond1 in subject_data and cond2 in subject_data:
                    # Stack the data for both conditions
                    stacked_data = np.vstack((subject_data[cond1], subject_data[cond2]))

                    # Append the reshaped data to X
                    X.append(stacked_data)

                    # Append the condition label for each pseudo-trial: 1 for cond1, 0 for cond2
                    n_pseudotrials_cond1 = subject_data[cond1].shape[0]
                    n_pseudotrials_cond2 = subject_data[cond2].shape[0]
                    y.extend([1] * n_pseudotrials_cond1)  # Label for cond1 is 1
                    y.extend([0] * n_pseudotrials_cond2)  # Label for cond2 is 0

                    # Append the subject identifier for each pseudo-trial
                    groups.extend([subject] * (n_pseudotrials_cond1 + n_pseudotrials_cond2))
                else:
                    print(f"Warning: Missing data for conditions {cond1} or {cond2} for subject {subject}")

            # Check if there is valid data for the current pair
            if X:
                # Convert lists to numpy arrays
                X = np.vstack(X)
                y = np.array(y)
                groups = np.array(groups)

                # Standardize the features
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Initialize the base classifiers
                base_classifiers = [
                    ('svm', SVC(kernel='linear', probability=True)),
                    ('lda', LDA()),
                    ('gnb', GaussianNB())
                ]
                
                # Initialize the stacking classifier with Logistic Regression as the meta-classifier
                stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())

                # Cross-validation: leave-one-subject-out
                auc_scores_for_pair = []
                for train_idx, test_idx in logo.split(X, y, groups):
                    # Get the subject being tested in this fold
                    test_subject = groups[test_idx][0]
                    print(f"Testing subject: {test_subject}")

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Apply PCA if specified (based on variance threshold)
                    if variance_threshold is not None:
                        pca = PCA(n_components=variance_threshold)
                        X_train = pca.fit_transform(X_train)
                        X_test = pca.transform(X_test)

                    # Train the stacking classifier
                    stacking_clf.fit(X_train, y_train)

                    # Predict probabilities for the test set
                    y_pred_prob = stacking_clf.predict_proba(X_test)[:, 1]

                    # Compute ROC-AUC score for the current fold
                    auc = roc_auc_score(y_test, y_pred_prob)
                    auc_scores_for_pair.append(auc)

                # Average the AUC scores for this condition pair
                mean_auc = np.mean(auc_scores_for_pair)
                auc_scores[(cond1, cond2)] = mean_auc

                # Save auc_scores_for_pair for this pair
                auc_scores_for_all_pairs[(cond1, cond2)] = auc_scores_for_pair

                # Print the AUC score for the current pair
                print(f"AUC score for {cond1} vs {cond2}: {mean_auc}")

                # Access the first key-value pair directly
                _, first_value = list(auc_scores_for_all_pairs.items())[0]

                # Get the length of the list (i.e., number of AUC scores for the first pair)
                print(f"AUC array for {cond1} vs {cond2} is of length: {len(first_value)}")

            else:
                print(f"Warning: No valid data found for {cond1} vs {cond2}")
    
    # Save both auc_scores and auc_scores_for_all_pairs
    np.save(f"{RSA_wd}/{save_filename}_across", auc_scores)
    np.save(f"{RSA_wd}/{save_filename}_within", auc_scores_for_all_pairs)

    return auc_scores


def process_condition_pair(cond1, cond2, data_dict, logo, variance_threshold):
    """
    Process a single pair of conditions and compute the AUC score.
    
    Parameters:
    - cond1, cond2 (str): Conditions to compare.
    - data_dict (dict): Dictionary with subjects as keys and their corresponding data as values.
    - logo (LeaveOneGroupOut): Leave-one-subject-out cross-validation object.
    - variance_threshold (float): Variance threshold for PCA.
    
    Returns:
    - (tuple): Condition pair, AUC score, and the list of individual AUC scores for the pair.
    """
    auc_scores_for_pair = []

    # Initialize variables for stacking data and labels
    X = []
    y = []
    groups = []

    # Iterate over subjects and stack data for the two conditions
    for subject, subject_data in data_dict.items():
        if cond1 in subject_data and cond2 in subject_data:
            # Stack the data for both conditions
            stacked_data = np.vstack((subject_data[cond1], subject_data[cond2]))

            # Append the reshaped data to X
            X.append(stacked_data)

            # Append the condition label for each pseudo-trial: 1 for cond1, 0 for cond2
            n_pseudotrials_cond1 = subject_data[cond1].shape[0]
            n_pseudotrials_cond2 = subject_data[cond2].shape[0]
            y.extend([1] * n_pseudotrials_cond1)  # Label for cond1 is 1
            y.extend([0] * n_pseudotrials_cond2)  # Label for cond2 is 0

            # Append the subject identifier for each pseudo-trial
            groups.extend([subject] * (n_pseudotrials_cond1 + n_pseudotrials_cond2))
        else:
            print(f"Warning: Missing data for conditions {cond1} or {cond2} for subject {subject}")

    # Check if there is valid data for the current pair
    if X:
        # Convert lists to numpy arrays
        X = np.vstack(X)
        y = np.array(y)
        groups = np.array(groups)

        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Initialize the base classifiers
        base_classifiers = [
            ('svm', SVC(kernel='linear', probability=True)),
            ('lda', LDA()),
            ('gnb', GaussianNB())
        ]
        
        # Initialize the stacking classifier with Logistic Regression as the meta-classifier
        stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=LogisticRegression())

        # Cross-validation: leave-one-subject-out
        for train_idx, test_idx in logo.split(X, y, groups):
            # Get the subject being tested in this fold
            test_subject = groups[test_idx][0]
            print(f"Testing subject: {test_subject}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply PCA if specified (based on variance threshold)
            if variance_threshold is not None:
                pca = PCA(n_components=variance_threshold)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train the stacking classifier
            stacking_clf.fit(X_train, y_train)

            # Predict probabilities for the test set
            y_pred_prob = stacking_clf.predict_proba(X_test)[:, 1]

            # Compute ROC-AUC score for the current fold
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores_for_pair.append(auc)

        # Average the AUC scores for this condition pair
        mean_auc = np.mean(auc_scores_for_pair)
        return (cond1, cond2, mean_auc, auc_scores_for_pair)
    else:
        print(f"Warning: No valid data found for {cond1} vs {cond2}")
        return (cond1, cond2, None, None)


def compute_pairwise_rsa_parallel(data_dict, conditions, variance_threshold, save_filename):
    """
    Perform RSA (pairwise binary classification) for each condition pair, and compute ROC-AUC scores.
    
    Parameters:
    - data_dict (dict): Dictionary with subjects as keys and their corresponding data as values.
    - conditions (list): List of conditions to compare.
    - save_filename (str): Name of file containing auc_scores.

    Returns:
    - auc_scores (dict): Dictionary with condition pairs as keys and their corresponding AUC scores as values.
    """
    
    auc_scores = {}
    auc_scores_for_all_pairs = {}  # Dictionary to store auc_scores_for_pair for each condition pair
    logo = LeaveOneGroupOut()

    # Create a list of condition pairs
    condition_pairs = [(cond1, cond2) for i, cond1 in enumerate(conditions) for j, cond2 in enumerate(conditions) if i < j]

    # Parallelize the condition pair processing
    results = Parallel(n_jobs=-1)(delayed(process_condition_pair)(cond1, cond2, data_dict, logo, variance_threshold) for cond1, cond2 in condition_pairs)

    # Collect the results into a dictionary and store auc_scores_for_pair
    for cond1, cond2, auc, auc_scores_for_pair in results:
        if auc is not None:
            auc_scores[(cond1, cond2)] = auc
            auc_scores_for_all_pairs[(cond1, cond2)] = auc_scores_for_pair  # Save the auc_scores_for_pair for each condition pair
            print(f"AUC score for {cond1} vs {cond2}: {auc}")

            # Access the first key-value pair directly
            _, first_value = list(auc_scores_for_all_pairs.items())[0]

            # Get the length of the list (i.e., number of AUC scores for the first pair)
            print(f"AUC array for {cond1} vs {cond2} is of length: {len(first_value)}")

        else:
            print(f"Warning: No valid data for {cond1} vs {cond2}")

    # Save both the auc_scores and auc_scores_for_all_pairs
    np.save(f"{RSA_wd}/{save_filename}_across", auc_scores)
    np.save(f"{RSA_wd}/{save_filename}_within", auc_scores_for_all_pairs)

    return auc_scores


# Helper function to categorize condition into Food or Non-food group
def get_condition_group(condition_name):
    if 'food' in condition_name.lower():
        return 'Food'
    else:
        return 'Non-food'

 # Helper function to extract lag duration and presentation from the condition name
def extract_lag_and_presentation(condition_name):
    # Extract lag duration (short, medium, long)
    lag = None
    for duration in ['short', 'medium', 'long']:
        if duration in condition_name.lower():
            lag = duration
            break
    
    # Extract presentation (rep1, rep2)
    presentation = None
    for rep in ['rep1', 'rep2']:
        if rep in condition_name.lower():
            presentation = rep
            break
    
    return lag, presentation


def plot_confusion_matrix_with_stats(auc_arrays, baseline_arrays, conditions, condition_labels=None, title1=None, title2=None, alpha=0.05):
    
    # Assuming auc_arrays and baseline_arrays are dictionaries with lists/arrays of values for each condition pair
    # Initialize matrix and results
    matrix = np.zeros((len(conditions), len(conditions)))  # Assuming conditions is a list of condition names
    significant_results = []
    all_results = []

    # Fill the AUC matrix with the pairwise AUC scores and perform Wilcoxon tests
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i >= j:
                continue  # Don't repeat pairs
            
            # Get AUC score from auc_arrays (assumed)
            auc_score = np.mean(auc_arrays.get((cond1, cond2), auc_arrays.get((cond2, cond1), [])))

            matrix[i, j] = auc_score
            matrix[j, i] = auc_score  # AUC is symmetric
            
            # Get the baseline score for this pair
            baseline_score = np.mean(baseline_arrays.get((cond1, cond2), baseline_arrays.get((cond2, cond1), [])))
            
            # Get the list of values for the pair from auc_arrays and baseline_arrays
            auc_values = auc_arrays.get((cond1, cond2), auc_arrays.get((cond2, cond1), []))
            baseline_values = baseline_arrays.get((cond1, cond2), baseline_arrays.get((cond2, cond1), []))
            
            # Check if there are enough values to run the test (they should be the same length)
            if len(auc_values) == len(baseline_values) and len(auc_values) > 0:
                # Run the Wilcoxon test on the differences between AUC and baseline values
                stat, p_value = wilcoxon(auc_values, baseline_values, zero_method='zsplit', alternative='greater')

                # Append results
                all_results.append(((cond1, cond2), auc_score, baseline_score, p_value))

                # Store if significant
                alpha = 0.05  # Adjust for your significance level
                if p_value < alpha and stat > 0:
                    significant_results.append(((cond1, cond2), auc_score, baseline_score, p_value))

    # Perform MDS to reduce the dimensionality of the AUC matrix
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(matrix)  # 1 - AUC to get dissimilarity

    # If custom labels are provided, use them, otherwise, use the original condition names
    labels_to_use = condition_labels if condition_labels else conditions

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns_heatmap = sns.heatmap(matrix, annot=False, fmt=".3f", cmap="coolwarm", xticklabels=labels_to_use, yticklabels=labels_to_use)

    # Add color bar label to heatmap
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label("Decodability (ROC-AUC Score)")

    plt.title(title1, fontdict={'fontsize': 14})
    plt.show()

    # Define custom colors for the two groups
    food_color = '#f4ccccff'  # Light pinkish for 'food'
    non_food_color = '#a8e6e3ff'  # Light turquoise for 'positive+neutral' (Non-food)

    # Assign each condition to a group based on its position in labels_to_use
    condition_to_group = {}
    for i, label in enumerate(labels_to_use):
        if i < 6:  # First 6 labels are 'food'
            group = 'Food'
        else:  # Last 12 labels are 'positive+neutral'
            group = 'Non-food'
        condition_to_group[label] = group

    # Create a dictionary to map groups to unique colors
    group_to_color = {
        'Food': food_color,
        'Non-food': non_food_color
    }

    # Plot MDS results (2D) with different colors for each group (food and positive+neutral)
    plt.figure(figsize=(9, 8))

    # Assign a color to each point based on its group
    for i, label in enumerate(labels_to_use):
        group = condition_to_group[label]  # Use the label to get the group
        color = group_to_color[group]  # Directly get the color from the group_to_color dictionary
        plt.scatter(mds_coords[i, 0], mds_coords[i, 1], c=color, marker='o', s=300, edgecolor='black', linewidth=1)

    # Annotate the points with condition names
    for i, label in enumerate(labels_to_use):
        plt.annotate(label, (mds_coords[i, 0], mds_coords[i, 1]), fontsize=14, ha='right')

    # Add lines connecting significant condition pairs (food vs non-food only) with same lag duration and presentation
    for cond_pair, auc_score, baseline_score, p_value in significant_results:
        cond1, cond2 = cond_pair
        i1, i2 = conditions.index(cond1), conditions.index(cond2)
        
        # Determine the group for each condition
        group1 = get_condition_group(cond1)
        group2 = get_condition_group(cond2)
        
        # Check if one condition is from 'food' and the other from 'non-food'
        if group1 == 'Food' and group2 == 'Non-food' or group1 == 'Non-food' and group2 == 'Food':
            # Extract lag duration and presentation for each condition
            _, presentation1 = extract_lag_and_presentation(cond1)
            _, presentation2 = extract_lag_and_presentation(cond2)
            
            # Only plot if the lag duration and presentation are the same
            if presentation1 == presentation2:
                # Plot line between significant pairs (food vs non-food only) with the same lag and presentation
                plt.plot([mds_coords[i1, 0], mds_coords[i2, 0]], 
                         [mds_coords[i1, 1], mds_coords[i2, 1]], 
                         color='black', lw=1, linestyle=':', alpha=0.5)  # Lighter line with alpha

    plt.title(title2, fontdict={'fontsize': 14})
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(False)
    plt.show()