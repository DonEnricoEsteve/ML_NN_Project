from utils import *

# def reshape_data(data_dict, conditions):
#     """
#     Reshape data from multiple subjects and conditions for classification.

#     Parameters:
#     - data_dict (dict): Keys are subject identifiers (e.g., "subject1") and values are dictionaries,
#                          with condition names as keys and 2D numpy arrays as values (after PCA).
#     - conditions (list of str): Names of conditions to include.

#     Returns:
#     - X (2D np.ndarray): Array of shape (total_pseudotrials, n_components) where n_components is the number of PCA components.
#     - y (1D np.ndarray): Array of shape (total_pseudotrials,), containing labels for each pseudo-trial.
#     - groups (1D np.ndarray): Array of shape (total_pseudotrials,), containing subject identifiers for each pseudo-trial.
#     """
    
#     X, y, groups = [], [], []  # Initialize lists to store reshaped data, labels, and groups
    
#     # Iterate over subjects and their data
#     for subject, subject_data in data_dict.items():
#         # Iterate over conditions and their indices
#         for cond_idx, condition in enumerate(conditions):
#             # Check if the condition exists for the current subject
#             if condition in subject_data:
#                 data = subject_data[condition] 
#                 n_pseudotrials = data.shape[0]  # Number of pseudo-trials

#                 # No need to reshape the data since it's already in the shape (n_pseudotrials, n_components)
#                 X.append(data)

#                 # Append the condition label for each pseudo-trial
#                 y.extend([cond_idx] * n_pseudotrials)

#                 # Append the subject identifier for each pseudo-trial
#                 groups.extend([subject] * n_pseudotrials)

#     # Convert lists to numpy arrays and return
#     return np.vstack(X), np.array(y), np.array(groups)


def reshape_data(data_dict, conditions):
    """
    Reshape data from multiple subjects and conditions for classification, and balance the classes
    by randomly sampling half of the larger class for each subject.
    
    Parameters:
    - data_dict (dict): Keys are subject identifiers (e.g., "subject1") and values are dictionaries,
                         with condition names as keys and 2D numpy arrays as values (after PCA).
    - conditions (list of str): Names of conditions to include.
    
    Returns:
    - X (2D np.ndarray): Array of shape (total_pseudotrials, n_components) where n_components is the number of PCA components.
    - y (1D np.ndarray): Array of shape (total_pseudotrials,), containing labels for each pseudo-trial.
    - groups (1D np.ndarray): Array of shape (total_pseudotrials,), containing subject identifiers for each pseudo-trial.
    """
    
    X, y, groups = [], [], []  # Initialize lists to store reshaped data, labels, and groups
    
    # Iterate over subjects and their data
    for subject, subject_data in data_dict.items():
        # Collect data by condition for the current subject
        condition_data = {condition: [] for condition in conditions}
        condition_labels = {condition: [] for condition in conditions}
        condition_groups = {condition: [] for condition in conditions}
        
        # Collect the data for each condition and subject
        for cond_idx, condition in enumerate(conditions):
            if condition in subject_data:
                data = subject_data[condition]
                n_pseudotrials = data.shape[0]
                
                # Store the data, labels, and groups for each condition
                condition_data[condition].append(data)
                condition_labels[condition].extend([cond_idx] * n_pseudotrials)
                condition_groups[condition].extend([subject] * n_pseudotrials)
        
        # For each condition, downsample to the minimum number of pseudo-trials per subject
        min_pseudotrials = min(len(condition_labels[condition]) for condition in conditions)
        
        for condition in conditions:
            n_pseudotrials = len(condition_labels[condition])
            
            if n_pseudotrials > min_pseudotrials:
                # Concatenate the data for this condition to a single array
                concatenated_data = np.concatenate(condition_data[condition], axis=0)
                
                # Randomly sample to match the minimum pseudo-trials for this subject
                indices = np.random.choice(concatenated_data.shape[0], min_pseudotrials, replace=False)
                
                # Downsample the data, labels, and groups
                condition_data[condition] = concatenated_data[indices]
                condition_labels[condition] = [condition_labels[condition][i] for i in indices]
                condition_groups[condition] = [condition_groups[condition][i] for i in indices]
            
            # Append the data, labels, and groups for this subject and condition
            X.append(condition_data[condition])
            y.extend(condition_labels[condition])
            groups.extend(condition_groups[condition])
    
    # Convert the lists into numpy arrays for the final output
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    groups = np.array(groups)
    
    return X, y, groups


def perform_general_decoding(base_dir, tasks, classification_type, output_filename, 
                             variance_threshold=None, shuffle_labels=False):
    """
    This function runs classification for both binary and multi-class conditions.
    
    Parameters:
    - base_dir (str): Directory where the subject data is stored
    - tasks (dict): Dictionary of tasks to be decoded
    - classification_type (str): 'binary' or 'multi' to specify the type of classification
    - output_filename (str): Filename for the results
    - variance_threshold (float, optional): Variance threshold for PCA. If None, PCA is not applied.
    - shuffle_labels (bool, optional): Whether to shuffle the labels before training the model. Default is False.
    """

    # List to store accuracies for each categorization set
    list_accuracies = []

    # Loop through each condition set (binary or multi)
    for task_key, task_set in tasks.items():
        print(f"Running with categorization: {task_key} - {task_set}")

        # Load the data for the current categorization set
        subjects_data = {}
        for subject in os.listdir(base_dir):
            subject_dir = os.path.join(base_dir, subject)
            if os.path.isdir(subject_dir):
                subject_data = {}
                for condition in task_set:  # Use the current categorization set
                    condition_file = os.path.join(subject_dir, f'{condition}.npy')
                    if os.path.exists(condition_file):
                        subject_data[condition] = np.load(condition_file)
                subjects_data[subject] = subject_data

        # Prepare the data
        X, y, groups = reshape_data(subjects_data, task_set)

        # Define the base classifiers
        base_classifiers = [
            ('svm', SVC(probability=True, random_state=42)),
            ('lda', LDA()),
            ('gnb', GaussianNB())
        ]

        # Final classifier (Logistic Regression) with default parameters
        final_classifier = LogisticRegression(random_state=42)

        # Create the stacking classifier without hyperparameter optimization
        stacking_clf = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=final_classifier
        )

        # Leave-One-Subject-Out Cross Validation
        logo = LeaveOneGroupOut()
        accuracies = []

        for train_idx, test_idx in logo.split(X, y, groups):
            # Extract the current test subject
            test_subject = groups[test_idx][0]
            print(f"Testing on subject: {test_subject}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Apply PCA if specified (based on variance threshold)
            if variance_threshold is not None:
                pca = PCA(n_components=variance_threshold)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                print(f"Applied PCA with variance threshold {variance_threshold * 100}%.")
                print(f"Number of components retained: {pca.n_components_}")
            
            # Shuffle labels if the flag is True
            if shuffle_labels:
                y_train = shuffle(y_train, random_state=42)
                print("Labels shuffled.")

            print(f"X_train shape:{X_train.shape}")
            print(f"X_test shape:{X_test.shape}")

            # Build the stacking classifier with default base models
            stacking_clf.fit(X_train, y_train)

            # Test the classifier
            y_pred = stacking_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"Subject {test_subject} accuracy: {accuracy:.3f}\n")

        # Append accuracies for the current categorization set
        list_accuracies.append(accuracies)

        # Overall accuracy for the current set
        print(f"Mean accuracy for {task_key}: {np.mean(accuracies):.3f}")

    # Save the list of accuracies to a file
    np.save(f"{results_dir}/{output_filename}_accuracies_{classification_type}.npy", list_accuracies)


def helper_evaluate_decoding(accuracies, condition_name, chance_level):
    """
    Evaluates the decoding performance for a given condition by calculating the mean accuracy,
    performing the Wilcoxon Signed-Rank test, and printing the statistical results.

    Parameters:
    - accuracies (list or np.array): A list or array containing the accuracy values for the condition.
    - condition_name (str): The name of the condition being evaluated (e.g., the categorization label).
    - chance_level (float): The chance level (or expected accuracy under random chance) for comparison.
    """
    
    mean_accuracy = np.mean(accuracies)
    print(f"Running with categorization: {condition_name}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print(f"Chance level: {chance_level}")

    # Perform Wilcoxon Signed-Rank Test (nonparametric test)
    stat, p_value = wilcoxon(np.array(accuracies) - chance_level)
    
    print(f"Wilcoxon Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpretation of significance
    if p_value < 0.05:
        print("The mean accuracy is significantly greater than chance level (p < 0.05).")
    else:
        print("The mean accuracy is not significantly different from chance level (p ≥ 0.05)")

    # Comparison of mean accuracy to chance level
    if mean_accuracy > chance_level:
        print("The model performs better than random chance.")
    else:
        print("The model performs worse than random chance.")

    print("\n" + "="*50 + "\n")  # Separator for readability


def evaluate_decoding(list_binary_accuracies, list_multi_accuracies,
                      binary_tasks, multiclass_tasks,
                      binary_chance_levels, multiclass_chance_levels):
    """
    Evaluates the decoding performance for both binary and multi-class classification conditions. 
    It calculates the mean accuracy for each condition, performs a Wilcoxon Signed-Rank test, 
    and compares the results with the respective chance levels.

    Parameters:
    - list_binary_accuracies (list of np.array): A list containing arrays of binary classification accuracies for each condition set.
    - list_multi_accuracies (list of np.array): A list containing arrays of multi-class classification accuracies for each condition set.
    - binary_tasks (dict): A dictionary where keys are condition names and values are the corresponding binary classification conditions.
    - multiclass_tasks (dict): A dictionary where keys are condition names and values are the corresponding multi-class classification conditions.
    - binary_chance_levels (dict): A dictionary containing the chance level for each binary condition.
    - multiclass_chance_levels (dict): A dictionary containing the chance level for each multi-class condition.    
    """
    
    # Evaluate Binary Classification Results
    print("Evaluating Binary Classification...\n")

    # Ensure binary_conditions_set is iterated over correctly (you can also use zip if you need to match binary_conditions_set with list_binary_accuracies)
    for condition_name, binary_conditions_set in binary_tasks.items():
        # Get the first list of accuracies (assuming list_binary_accuracies is a list of lists)
        b_acc = list_binary_accuracies[0]  # Get the first list for the current categorization set
        list_binary_accuracies = list_binary_accuracies[1:]  # Remove the first list after using it

        # Get the chance level for the current condition
        chance_level = binary_chance_levels.get(condition_name, 0.50)

        # Evaluate decoding with the extracted binary accuracy and chance level
        helper_evaluate_decoding(b_acc, condition_name, chance_level)

    # Evaluate Multi-Class Classification Results
    print("Evaluating Multi-Class Classification...\n")

    # Ensure multiclass_conditions_set is iterated over correctly (you can also use zip if you need to match multiclass_conditions_set with list_multi_accuracies)
    for condition_name, multi_conditions_set in multiclass_tasks.items():
        # Get the first list of accuracies (assuming list_multi_accuracies is a list of lists)
        m_acc = list_multi_accuracies[0]  # Get the first list for the current categorization set
        list_multi_accuracies = list_multi_accuracies[1:]  # Remove the first list after using it

        # Get the chance level for the current condition
        chance_level = multiclass_chance_levels.get(condition_name, 0.33)

        # Evaluate decoding with the extracted multi-class accuracy and chance level
        helper_evaluate_decoding(m_acc, condition_name, chance_level)

