from utils import *

def preprocess_helper(input_dir, output_dir, process_func, suffix='processed', **process_func_args):
    """
    Process all .npy files in each subject directory (sub_XXX), apply the provided processing function,
    and save the results to the output directory.

    Parameters:
    - input_dir (str): The directory containing subject directories (sub_XXX).
    - output_dir (str): The directory where the processed data will be saved.
    - process_func (function): The processing function to be applied to the data.
    - suffix (str): The suffix to be added to the processed files (default is 'processed').
    - **process_func_args: Additional arguments to pass to the processing function.
    """

    # Get a sorted list of subject directories
    subject_dirs = sorted([sub_dir for sub_dir in os.listdir(input_dir) 
                           if sub_dir.startswith('sub_') and os.path.isdir(os.path.join(input_dir, sub_dir))])

    # Iterate through each subject directory
    for sub_dir in subject_dirs:
        sub_dir_path = os.path.join(input_dir, sub_dir)
        
        # Create the corresponding subfolder in the output directory
        output_subfolder = os.path.join(output_dir, sub_dir)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Iterate through each .npy file in the subject directory
        for file_name in os.listdir(sub_dir_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(sub_dir_path, file_name)
                
                # Load the data
                data = np.load(file_path)
                
                # Apply the provided processing function to the data
                processed_data = process_func(data, **process_func_args)
                
                # Save the processed data to the respective subfolder in the output directory
                processed_file_name = file_name.replace('.npy', f'_{suffix}.npy')
                processed_file_path = os.path.join(output_subfolder, processed_file_name)
                np.save(processed_file_path, processed_data)

                # Print for tracking purposes
                print(f"Processed and saved: {processed_file_name} in {output_subfolder}")

# def decimate_data(data, bin_size_ms=20, sampling_rate_hz=1000):
#     """
#     Takes a 3D numpy array and reduces the length of the last axis (time) by averaging.

#     Parameters:
#     - data (np.ndarray): A 3D matrix of shape (n_epochs, n_channels, n_times)
#     - bin_size_ms (int): The length of time (in ms) within which each time point is consecutively averaged. Default is 20 ms.
#     - sampling_rate_hz (int): The sampling frequency (in Hz). Default is 1000 Hz

#     Returns: 
#     A 3D numpy array of shape n_epochs, n_channels, n_times with the last axis reduced.
#     """

#     data = np.abs(data)

#     bin_size_samples = int(bin_size_ms * sampling_rate_hz / 1000)
    
#     # Get the length of each axis of the 3D matrix
#     n_epochs, n_channels, n_times = data.shape
    
#     # Calculate how many time points to trim
#     trim_samples = n_times % bin_size_samples
#     if trim_samples != 0:
#         # Trim the last `trim_samples` time points - Last 9 samples
#         data = data[:, :, :-trim_samples]
    
#     # Now the number of time points is divisible by the bin size
#     n_times_trimmed = data.shape[2]
    
#     # Reshape and average the data over bins
#     reshaped_data = data.reshape(n_epochs, n_channels, n_times_trimmed // bin_size_samples, bin_size_samples)
#     decimated_data = reshaped_data.mean(axis=-1)
    
#     return decimated_data

# def calculate_pseudo_trials(data, n_groups=10):
#     """
#     Function that computes pseudo-trials from raw (single) trials.
#     For each condition, raw trials (n=~20-30) were randomly placed in 5 groups then averaged. 

#     Parameters:
#     - data (np.ndarray): A 3D matrix of shape (n_epochs, n_channels, n_times)
#     - n_groups (int): Number of pseudo-trials to be defined. Default value is 5. Must not be larger than 20.

#     Returns: A 3D numpy array of shape n_epochs, n_channels, n_times with the first axis reduced to n_groups.
#     """
#     # Ensure n_groups is not larger than 20 (minimum number of single trials in a condition)
#     assert n_groups <= 20, f"n_groups must not be larger than 20, but got {n_groups}"

#     # Get the length of each axis of the 3D matrix
#     n_epochs, n_channels, n_times = data.shape

#     # Split the epochs into n_groups
#     group_size = n_epochs // n_groups
#     groups = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]
    
#     # Handle any remaining epochs (in case n_epochs is not divisible by n_groups)
#     if n_epochs % n_groups != 0:
#         groups[-1] = np.concatenate([groups[-1], np.arange(n_groups * group_size, n_epochs)])

#     # Step 2: Average epochs within each group to create pseudo-trials
#     group_pseudo_trials = []
#     for group in groups:
#         pseudo_trial = np.mean(data[group, :, :], axis=0)  # Average over the epochs in the group
#         group_pseudo_trials.append(pseudo_trial)

#     # Step 3: Stack the pseudo-trials into a 3D array (n_groups x n_channels x n_times)
#     return np.stack(group_pseudo_trials, axis=0)

# def perform_pca_all_time(data, n_components=10):
#     """
#     Perform PCA on a 3D array across all time points for each pseudo-trial independently.
    
#     Parameters:
#     - data (np.ndarray): Input data of shape (n_pseudotrials, n_channels, n_timepoints).
#     - n_components (int or None): Number of PCA components to retain. If None, all components are retained. Default value is 10.
    
#     Returns:
#     - pca_transformed_data (np.ndarray): Transformed data of shape (n_pseudotrials, n_components, n_timepoints).
#     - pca_models (list): List of PCA objects for each pseudo-trial (useful for inspecting PCA components).
#     """
#     n_pseudotrials, n_channels, n_timepoints = data.shape

#     # Prepare to store the transformed data and PCA models
#     pca_transformed_data = []
#     pca_models = []

#     # Apply PCA for each pseudo-trial independently
#     for trial in range(n_pseudotrials):
#         # Flatten the data for the current pseudo-trial (n_channels x n_timepoints -> n_features)
#         trial_data = data[trial, :, :].reshape(n_channels, n_timepoints).T
        
#         # Apply PCA
#         pca = PCA(n_components=n_components)
#         transformed = pca.fit_transform(trial_data)
        
#         # Store the transformed data and PCA model
#         pca_transformed_data.append(transformed.T)  # Transpose back to (n_components x n_timepoints)
#         # pca_models.append(pca)
    
#     # Stack transformed data back into a 3D array (n_pseudotrials x n_components x n_timepoints)
#     pca_transformed_data = np.stack(pca_transformed_data, axis=0)

#     return pca_transformed_data

def calculate_pseudo_trials_with_pca(data, n_groups=None, n_components=None):
    """
    Function that computes pseudo-trials from raw (single) trials and performs PCA to reduce the number of channels.
    For each condition, raw trials (n=~20-30) were randomly placed in 5 groups then averaged.
    
    Parameters:
    - data (np.ndarray): A 3D matrix of shape (n_epochs, n_channels, n_times)
    - n_groups (int): Number of pseudo-trials to be defined. Default value is 5. Must not be larger than 20.
    - n_components (int): Number of PCA components to retain. Default is 5.

    Returns:
    - A 3D numpy array of shape (n_groups, n_components) after averaging over time and performing PCA.
    """
    # Ensure n_groups is not larger than 20 (minimum number of single trials in a condition)
    assert n_groups <= 20, f"n_groups must not be larger than 20, but got {n_groups}"

    # Get the length of each axis of the 3D matrix
    n_epochs, n_channels, n_times = data.shape

    # Step 1: Average the data over the time axis (n_times)
    data_avg_time = np.mean(data, axis=2)  # Resulting shape: (n_epochs, n_channels)

    # Step 2: Split the epochs into n_groups
    group_size = n_epochs // n_groups
    groups = [np.arange(i * group_size, (i + 1) * group_size) for i in range(n_groups)]
    
    # Handle any remaining epochs (in case n_epochs is not divisible by n_groups)
    if n_epochs % n_groups != 0:
        groups[-1] = np.concatenate([groups[-1], np.arange(n_groups * group_size, n_epochs)])

    # Step 3: Average epochs within each group to create pseudo-trials
    group_pseudo_trials = []
    for group in groups:
        pseudo_trial = np.mean(data_avg_time[group, :], axis=0)  # Average over the epochs in the group
        group_pseudo_trials.append(pseudo_trial)

    # Step 4: Stack the pseudo-trials into a 2D array (n_groups x n_channels)
    pseudo_trials = np.stack(group_pseudo_trials, axis=0)  # Shape: (n_groups, n_channels)

    # # Step 5: Apply PCA to reduce the number of channels (n_components)
    # pca = PCA(n_components=n_components)
    # pca_transformed = pca.fit_transform(pseudo_trials)  # Shape: (n_groups, n_components)
    # return pca_transformed

    return pseudo_trials

def check_output(s):
    n_psuedotrials, n_components = np.load(s).shape
    print(f"Number of features is: {n_components}")