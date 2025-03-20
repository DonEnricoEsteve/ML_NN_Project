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
                

def calculate_pseudo_trials(data, n_groups):
    """
    Function that computes pseudo-trials from raw (single) trials and performs PCA to reduce the number of channels.
    For each condition, raw trials (n=~20-30) were randomly placed in 5 groups then averaged.
    
    Parameters:
    - data (np.ndarray): A 2DD matrix of shape (n_epochs, n_channels)
    - n_groups (int): Number of pseudo-trials to be defined. Default value is 5. Must not be larger than 20.
    - n_components (int): Number of PCA components to retain. Default is 5.

    Returns:
    - A 3D numpy array of shape (n_groups, n_components) after averaging over time and performing PCA.
    """
    # Ensure n_groups is not larger than 20 (minimum number of single trials in a condition)
    assert n_groups <= 20, f"n_groups must not be larger than 20, but got {n_groups}"

    # Get the length of each axis of the 3D matrix
    n_epochs, n_channels, n_times = data.shape

    # # Take absolute value
    # data = np.abs(data)

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

    return pseudo_trials

def check_output(s):
    n_psuedotrials, n_components = np.load(s).shape
    print(f"Number of features is: {n_components}")