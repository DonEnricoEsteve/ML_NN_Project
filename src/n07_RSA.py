from utils import *

def RSA_convert_epochsFIF_to_npy(fif_input_directory: str, npy_output_directory: str, tmin: float = None, tmax: float = None):
    """
    Processes the epoched data files in the given directory, optionally crops the epochs to a time window of interest,
    and saves the concatenated numpy arrays for each condition.

    Parameters:
    - fif_input_directory (str): The path to the directory containing the epoched .fif files.
    - npy_output_directory (str): The path to the directory where the processed .npy arrays will be saved.
    - conds (list): List of condition names corresponding to each grouped set of conditions.
    - tmin (float, optional): The start time for cropping the epochs (in seconds). If None, no cropping is applied.
    - tmax (float, optional): The end time for cropping the epochs (in seconds). If None, no cropping is applied.
    """

    # Get a list of all subject files and saving folders
    subject_files = sorted(Path(fif_input_directory).glob("sub*"))
    save_folders = sorted(Path(npy_output_directory).glob("sub*"))

    # Ensure the lengths match before iterating
    if len(subject_files) != len(save_folders):
        print(f"Error: The number of files and saving directories do not match.")
        return

    # Process each subject's data
    for file, saving_folder in zip(subject_files, save_folders):
        # Create instance of Epochs class
        epochs = mne.read_epochs(file)

        # Optionally crop only the time of interest (if tmin and tmax are provided)
        if tmin is not None and tmax is not None:
            epochs_toi = epochs.crop(tmin=tmin, tmax=tmax)
        else:
            epochs_toi = epochs  # No cropping if tmin and tmax are None

        # Create a list of all conditions in the epoched data
        conditions = list(epochs_toi.event_id.keys())

        # Take the data and save it
        for con in conditions:
            con_npy = epochs_toi[con].get_data()
            
            # Replace '/' with '_' in the condition name
            con = con.replace('/', '_')
            
            savepath = os.path.join(saving_folder, con + '.npy')
            np.save(savepath, con_npy)

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

def compute_pairwise_rsa(data_dict, conditions, save_filename):
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
                    # Take the mean across the time dimension (axis=2) while keeping the channel dimension (axis=1)
                    mean_data_cond1 = np.mean(subject_data[cond1], axis=2)  # Shape: (n_pseudotrials, n_channels)
                    mean_data_cond2 = np.mean(subject_data[cond2], axis=2)  # Shape: (n_pseudotrials, n_channels)

                    # Stack the data for both conditions
                    stacked_data = np.vstack((mean_data_cond1, mean_data_cond2))

                    # Append the reshaped data to X
                    X.append(stacked_data)

                    # Append the condition label for each pseudo-trial: 1 for cond1, 0 for cond2
                    n_pseudotrials_cond1 = mean_data_cond1.shape[0]
                    n_pseudotrials_cond2 = mean_data_cond2.shape[0]
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

                    X_train, X_test = X[train_idx], X[test_idx] # (n_subjects*[n_trials1+n_trials2], n_components) (n_trials1+n_trials2, n_components)
                    y_train, y_test = y[train_idx], y[test_idx] # (n_subjects*[n_trials1+n_trials2], )             (n_trials1+n_trials2, )

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

                # Print the AUC score for the current pair
                print(f"AUC score for {cond1} vs {cond2}: {mean_auc}")

                np.save(f"{RSA_wd}/Results/{save_filename}", auc_scores)

            else:
                print(f"Warning: No valid data found for {cond1} vs {cond2}")

    return auc_scores

def plot_confusion_matrix_with_mds(auc_results, conditions, condition_labels=None, title1=None, title2=None):
    # Initialize a matrix to hold the pairwise AUC scores
    n_conditions = len(conditions)
    matrix = np.zeros((n_conditions, n_conditions))

    # Fill the AUC matrix with the pairwise AUC scores
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i >= j:
                continue  # Don't repeat pairs
            auc_score = auc_results.get((cond1, cond2), auc_results.get((cond2, cond1), 0))
            matrix[i, j] = auc_score
            matrix[j, i] = auc_score  # AUC is symmetric

    # Perform MDS to reduce the dimensionality of the AUC matrix
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(1 - matrix)  # 1 - AUC to get dissimilarity

    # If custom labels are provided, use them, otherwise, use the original condition names
    labels_to_use = condition_labels if condition_labels else conditions

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns_heatmap = sns.heatmap(matrix, annot=False, fmt=".3f", cmap="coolwarm", xticklabels=labels_to_use, yticklabels=labels_to_use)

    # Add color bar label to heatmap
    cbar = sns_heatmap.collections[0].colorbar
    cbar.set_label("1 — ROC-AUC Score")

    plt.title(title1, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.show()

    # Define colors for each group (category + presentation combination)
    n_groups = 6  # 6 groups: food_rep1, food_rep2, positive_rep1, positive_rep2, neutral_rep1, neutral_rep2
    colors = plt.cm.get_cmap('tab20', n_groups)  # Get a colormap with enough distinct colors

    # Assign each condition to a group based on its category and presentation (food_rep1, food_rep2, etc.)
    condition_to_group = {}
    for i, cond in enumerate(conditions):
        parts = cond.split('_')  # Split condition into parts (e.g., ['food', 'rep1', 'pca'])
        category = parts[1]  # 'food', 'positive', 'neutral'
        presentation = parts[2]  # 'rep1', 'rep2'
        group = f'{category}_{presentation}'  # Create a group like 'food_rep1'
        condition_to_group[labels_to_use[i]] = group

    # Create a dictionary to map groups to unique colors
    group_to_color = {group: colors(i) for i, group in enumerate(set(condition_to_group.values()))}

    # Plot MDS results (2D) with different colors for each group (category + presentation)
    plt.figure(figsize=(9, 8))

    # Assign a color to each point based on its group
    for i, label in enumerate(labels_to_use):
        group = condition_to_group[label]  # Use the label to get the group
        plt.scatter(mds_coords[i, 0], mds_coords[i, 1], c=[group_to_color[group]], marker='o', s=300, edgecolor='black', linewidth=1)

    # Annotate the points with condition names
    for i, label in enumerate(labels_to_use):
        plt.annotate(label, (mds_coords[i, 0], mds_coords[i, 1]), fontsize=14, ha='right')

    plt.title(title2, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

    