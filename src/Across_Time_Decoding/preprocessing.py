import numpy as np
from config import trials_to_retain, F_events_code

# Don
def generate_pseudo_trials(data: type[np.ndarray], n_groups: int = 10) -> type[np.ndarray]:
    """
    Function computes pseudo-trials from raw (single) trials.
    For each condition, raw trials (n=~20-30) were randomly placed in 10 groups then averaged. 

    Parameters:
    * data (np.ndarray): A 3D matrix of shape (n_epochs, n_channels, n_times)
    * n_groups (int): Number of pseudo-trials to be defined. Default value is 10. Must not be larger than 20.

    Returns (np.ndarray): A 3D numpy array of shape n_epochs, n_channels, n_times with the first axis reduced to n_groups.
    """
    # Ensure n_groups is not larger than 20 (minimum number of single trials in a condition)
    assert n_groups <= 20, f"n_groups must not be larger than 20, but got {n_groups}"

    # Get the length of each axis of the 3D matrix
    n_epochs, n_channels, n_times = data.shape

    # Split the epochs into n_groups
    group_size = n_epochs // n_groups
    groups = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]
    
    # Handle any remaining epochs (in case n_epochs is not divisible by n_groups)
    if n_epochs % n_groups != 0:
        groups[-1] = np.concatenate([groups[-1], np.arange(n_groups * group_size, n_epochs)])

    # Step 2: Average epochs within each group to create pseudo-trials
    group_pseudo_trials = []
    for group in groups:
        pseudo_trial = np.mean(data[group], axis=0)  # Average over the epochs in the group
        group_pseudo_trials.append(pseudo_trial)

    # Step 3: Stack the pseudo-trials into a 3D array (n_groups x n_channels x n_times)
    pseudo_trials = np.stack(group_pseudo_trials, axis=0)

    return pseudo_trials

def retain_events(data: type[np.ndarray], events_ids: type[np.ndarray], contrast_events_code: type[np.ndarray])-> tuple[type[np.ndarray], type[np.ndarray]]:
    """
    Function that retains only the relevant events for the contrast, according to the  

    Parameters:
    * data - size: n_psedotrials * n_channels * n_times array.
      Pseudo trials for all conditions.
    * events_ids - size: 1 * n_psedotrials (for all conditions). 
    * contrast_events_code - the event_ids to retain (depending on the contrast/decoding task).

    Returns: 
    tuple:
        1) rettained_array - a 3D numpy array of shape n_pseudotrials_retained, n_channels, n_times, only pseudo_trials relating to the conditions
           in the contrast are retained. 
        2) retained_event_id - a 1D numpy array of length n_pseudotrials_retained.
    """

    # Find the indices of all events in events_ids that match any of the contrast_events_code
    indices = np.nonzero(np.isin(events_ids, contrast_events_code))[0]
    
    # Extract the corresponding event ids
    retained_event_id = np.array(events_ids)[indices.astype(int)]
    
    # Extract the corresponding data for those event ids
    retained_array = data[indices.astype(int)]

    return (retained_array, retained_event_id)

def balance_helper(array, labels, binary_labels, larger_label_idx, smaller_label_idx):
    balanced_list = []  # To hold the balanced data
    balanced_labels = []  # To hold the corresponding binary labels

    # Add all trials from the smaller class
    balanced_list.append(array[smaller_label_idx])
    balanced_labels.append(binary_labels[smaller_label_idx])

    # Get the binary label value for the larger class
    bin_num = np.unique(binary_labels[larger_label_idx])
    
    # Calculate how many trials to retain from the larger class to match the number of trials in the smaller class,
    # scaled by a factor `trials_to_retain` - the factor from config.py which indicates the initial number of pseudo_trials
    n_trials_retain = trials_to_retain * (len(larger_label_idx) / len(smaller_label_idx))

    # For each label in the larger class, randomly sample a balanced number of trials
    for label in labels[larger_label_idx]:
        # Randomly select trials of the current label
        balanced_list.append(
            np.random.generator.choice(
                a=array[np.nonzero(label == labels)],
                size=n_trials_retain,
                replace=False
            )
        )
        # Assign the corresponding binary label
        balanced_labels.append([bin_num] * n_trials_retain)

    # Concatenate the balanced data and labels into final arrays
    balanced_array = np.vstack(balanced_list)
    balanced_labels = np.hstack(balanced_labels)

    return balanced_array, balanced_labels
    
def balance_trials(array: type[np.ndarray], labels: type[np.ndarray], binary_labels: type[np.ndarray]) -> tuple[type[np.ndarray], type[np.ndarray]]:
    """
    Function that balances trials in a binary contrast in case there is an imbalance.
    
    Parameters:
    * array - an array of all trials in the desired contrast.
    * binary labels - labels after creating the binary contrast
    * labels - original event_ids for each pseudo_trial 

    Returns: 
    1D tuple:
        1) A numpy array of shape n_trials, n_channels, n_times with the first axis having the balanced trials.
        2) A 1D numpy array of binary labels corresponding to the balanced pseudo_trials.
    """

    # Initialize the output as the original inputs (if already balanced, these will be returned as is)
    balanced_array, balanced_labels = array, binary_labels

    # Find indices of trials belonging to each binary class
    label_0_idx = np.nonzero(binary_labels == np.unique(binary_labels)[0])
    label_1_idx = np.nonzero(binary_labels == np.unique(binary_labels)[1])

    # If class 0 has fewer trials, balance by downsampling class 1
    if len(label_0_idx) < len(label_1_idx):
        balanced_array, balanced_labels = balance_helper(
            array=array,
            labels=labels,
            binary_labels=binary_labels,
            larger_label_idx=label_1_idx,
            smaller_label_idx=label_0_idx
        )

    # If class 1 has fewer trials, balance by downsampling class 0
    elif len(label_1_idx) < len(label_0_idx):
        balanced_array, balanced_labels = balance_helper(
            array=array,
            labels=labels,
            binary_labels=binary_labels,
            larger_label_idx=label_0_idx,
            smaller_label_idx=label_1_idx
        )

    return (balanced_array, balanced_labels)

# A wrapper for preprocessing the data:
def preprocess_data(epochs, trials_to_retain, contrast_events_code):
    
    X_psuedo_all = []
    labels_pseudo_all = []

    # Itterate over all conditions in dataset
    for cond_key, cond_value in epochs.event_id.items():

        # Average all trials of a condition to 10 pseudo trials
        x_pseudo = generate_pseudo_trials(data=epochs[cond_key].get_data(), n_groups=trials_to_retain)

        # Create a list of the event_id of the current condition repeated as many times as the number of pseudo trials.  
        label_pseudo = [cond_value]*len(x_pseudo) 

        # Concatenate pseudo trials and event_ids of all the conditions in the epochs data set.
        X_psuedo_all.append(x_pseudo)
        labels_pseudo_all.extend(label_pseudo)

    X_psuedo_all = np.concatenate(X_psuedo_all)   

    retained_array, retained_event_id = retain_events(data=X_psuedo_all, events_ids=labels_pseudo_all, contrast_events_code=contrast_events_code)

    # The food category will be always labeld 1 in this analysis
    binary_labels = np.where(np.isin(retained_event_id, F_events_code), 1, 0)

    balanced_array, balanced_event_ids = balance_trials(array=retained_array, labels=retained_event_id, binary_labels=binary_labels)

    return balanced_array, balanced_event_ids