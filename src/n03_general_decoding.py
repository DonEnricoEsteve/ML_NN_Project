from utils import *

def reshape_data(data_dict, conditions):
    """
    Reshape data from multiple subjects and conditions for classification.

    Parameters:
    - data_dict (dict): keys are subject identifiers (e.g., "subject1") and values are dictionaries, with condition names as keys and 3D numpy arrays as values. 
    - conditions (list of str): names of conditions to include.

    Returns:
    - X (2D np.ndarray): Array of shape (total_pseudotrials, n_features) where n_features = n_channels * n_times.
    - y (1D np.ndarray): Array of shape (total_pseudotrials,), containing labels for each pseudo-trial.
    - groups (1D np.ndarray): Array of shape (total_pseudotrials,), containing subject identifiers for each pseudo-trial.
    """

    X, y, groups = [], [], []  # Initialize lists to store reshaped data, labels, and groups

    # Iterate over subjects and their data
    for subject, subject_data in data_dict.items():
        # Iterate over conditions and their indices
        for cond_idx, condition in enumerate(conditions):
            # Check if the condition exists for the current subject
            if condition in subject_data:
                data = subject_data[condition]  # Get the data for the current condition
                n_pseudotrials = data.shape[0]  # Number of pseudo-trials

                # Reshape the data: (n_pseudotrials, n_channels * n_times)
                X.append(data.reshape(n_pseudotrials, -1))

                # Append the condition label for each pseudo-trial
                y.extend([cond_idx] * n_pseudotrials)

                # Append the subject identifier for each pseudo-trial
                groups.extend([subject] * n_pseudotrials)

    # Convert lists to numpy arrays and return
    return np.vstack(X), np.array(y), np.array(groups)

def perform_general_decoding(base_dir, tasks, classification_type, output_filename):
    """
    This function runs classification for both binary and multi-class conditions.
    
    Parameters:
    - base_dir (str): Directory where the subject data is stored
    - tasks (dict): Dictionary of tasks to be decoded
    - classification_type (str): 'binary' or 'multi' to specify the type of classification
    - output_filename (str): Filename for the results
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

            # Build the stacking classifier with default base models
            stacking_clf.fit(X_train, y_train)

            # Test the classifier
            y_pred = stacking_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(f"Subject {test_subject} accuracy: {accuracy:.2f}\n")

        # Append accuracies for the current categorization set
        list_accuracies.append(accuracies)

        # Overall accuracy for the current set
        print(f"Mean accuracy for {task_key}: {np.mean(accuracies):.2f}")

    # Save the list of accuracies to a file
    np.save(f"{results_dir}/list_{classification_type}_accuracies_{output_filename}.npy", list_accuracies) 

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
    for condition_name, binary_conditions_set in binary_tasks.items():
        b_acc = list_binary_accuracies.pop(0)  # Get the accuracies for the current categorization set
        chance_level = binary_chance_levels.get(condition_name, 0.50)
        helper_evaluate_decoding(b_acc, condition_name, chance_level)

    # Evaluate Multi-Class Classification Results
    print("Evaluating Multi-Class Classification...\n")
    for condition_name, multi_conditions_set in multiclass_tasks.items():
        m_acc = list_multi_accuracies.pop(0)  # Get the accuracies for the current categorization set
        chance_level = multiclass_chance_levels.get(condition_name, 0.33)
        helper_evaluate_decoding(m_acc, condition_name, chance_level)
