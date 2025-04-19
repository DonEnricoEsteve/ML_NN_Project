from utils import *

def convert_epochsFIF_to_npy(fif_input_directory: str, npy_output_directory: str, tmin: float, tmax: float):

    """
    Processes the epoched data files in the given directory, optionally crops the epochs to a time window of interest,
    and saves the concatenated numpy arrays for each condition.

    Parameters:
    - fif_input_directory (str): The path to the directory containing the epoched .fif files.
    - npy_output_directory (str): The path to the directory where the processed .npy arrays will be saved.
    - conds (list): List of condition names corresponding to each grouped set of conditions.
    - tmin (float): The start time for cropping the epochs (in seconds). If None, no cropping is applied.
    - tmax (float): The end time for cropping the epochs (in seconds). If None, no cropping is applied.
    """

    # Get a list of all subject files and saving folders
    subject_files = sorted(Path(fif_input_directory).glob("sub*"))
    save_folders = sorted(Path(npy_output_directory).glob("sub*"))

    # Ensure the lengths match before iterating
    if len(subject_files) != len(save_folders):
        print(f"Error: The number of files and saving directories do not match.")

    # Process each subject's data
    for file, saving_folder in zip(subject_files, save_folders):
        # Create instance of Epochs class
        epochs = mne.read_epochs(file)
        epochs_toi = epochs.crop(tmin=tmin, tmax=tmax)

        # Get list of conditions and replace slashes with underscores in the conditions list
        conditions = list(epochs_toi.event_id.keys())
        conditions_for_saving = [condition.replace('/', '_') for condition in conditions]

        # Iterate over conditions
        for condition, condition_for_saving in zip(conditions, conditions_for_saving):
            # Select epochs for the current condition
            condition_epochs = epochs_toi[condition]
            
            # Convert the epochs to a numpy array (data)
            condition_data = condition_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

            # Create a file path for saving the numpy array
            save_path = saving_folder / f"{condition_for_saving}.npy"
            
            # Save the condition data as a .npy file
            np.save(save_path, condition_data)

            print(f"Saved {condition_for_saving} data for {file.stem} to {save_path}")


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
    - data (np.ndarray): A 2D matrix of shape (n_epochs, n_channels)
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

    return pseudo_trials


def derive_class_labels(npy_IO_directory, suffix):
    """
    Derive new class labels from the base conditions.
    """

    # Get and sort the list of subject folders (ensure we're matching subdirectories)
    subject_files = sorted(glob.glob(os.path.join(npy_IO_directory, '*/')))  # Match all subdirectories
    save_folders = sorted(glob.glob(os.path.join(npy_IO_directory, '*/')))  # Match all subdirectories

    # Ensure the lengths match before iterating
    assert len(subject_files) == len(save_folders), "Number of files and saving directories do not match."

    # Process each subject
    for file, saving_folder in zip(subject_files, save_folders):
        # Get the directory name from the full file path
        subject_dir = os.path.dirname(file)

        # Define file paths for each condition
        original_categories = ['food', 'positive', 'neutral']
        lags = ['short', 'medium', 'long']
        presentations = ['1', '2']

        condition_files = {
            f'{category}_{lag}_{pres}': os.path.join(subject_dir, f'{category}_{lag}_rep{pres}_{suffix}.npy')
            for category in original_categories
            for lag in lags
            for pres in presentations
        }

        # Check if all files exist
        if all(os.path.exists(path) for path in condition_files.values()):
            # Load all condition files
            data = {key: np.load(path) for key, path in condition_files.items()}
            print(f"Loaded data for subject {file}")
        else:
            print(f"Missing .npy files for subject {file}, skipping...")
            continue

        # Stack the data for various conditions. This part of the code was not shortened for transparency
        # Food-related
        food = np.vstack([data['food_short_1'], data['food_medium_1'], data['food_long_1'], 
                          data['food_short_2'], data['food_medium_2'], data['food_long_2']])
        food_1 = np.vstack([data['food_short_1'], data['food_medium_1'], data['food_long_1']])
        food_2 = np.vstack([data['food_short_2'], data['food_medium_2'], data['food_long_2']])

        # Positive-related
        positive = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                          data['positive_short_2'], data['positive_medium_2'], data['positive_long_2']])
        positive_1 = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1']])
        positive_2 = np.vstack([data['positive_short_2'], data['positive_medium_2'], data['positive_long_2']])

        # Neutral-related
        neutral = np.vstack([data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1'], 
                          data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2']])
        neutral_1 = np.vstack([data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1']])
        neutral_2 = np.vstack([data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2']])

        # Non-food-related
        nonfood = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                             data['positive_short_2'], data['positive_medium_2'], data['positive_long_2'],
                             data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1'],
                             data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2']])
        nonfood_1 = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                               data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1']])
        nonfood_2 = np.vstack([data['positive_short_2'], data['positive_medium_2'], data['positive_long_2'], 
                               data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2']])
        
        # Salient-related
        salient = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                             data['positive_short_2'], data['positive_medium_2'], data['positive_long_2'],
                             data['food_short_1'], data['food_medium_1'], data['food_long_1'],
                             data['food_short_2'], data['food_medium_2'], data['food_long_2']])
        salient_1 = np.vstack([data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                               data['food_short_1'], data['food_medium_1'], data['food_long_1']])
        salient_2 = np.vstack([data['positive_short_2'], data['positive_medium_2'], data['positive_long_2'], 
                               data['food_short_2'], data['food_medium_2'], data['food_long_2']])
        
        # Control-related
        control = np.vstack([data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1'], 
                             data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2'],
                             data['food_short_1'], data['food_medium_1'], data['food_long_1'],
                             data['food_short_2'], data['food_medium_2'], data['food_long_2']])
        control_1 = np.vstack([data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1'], 
                               data['food_short_1'], data['food_medium_1'], data['food_long_1']])
        control_2 = np.vstack([data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2'], 
                               data['food_short_2'], data['food_medium_2'], data['food_long_2']])
        
        # Presentation-related
        pres_1 = np.vstack([data['food_short_1'], data['food_medium_1'], data['food_long_1'],
                            data['positive_short_1'], data['positive_medium_1'], data['positive_long_1'], 
                            data['neutral_short_1'], data['neutral_medium_1'], data['neutral_long_1']])
        pres_2 = np.vstack([data['food_short_2'], data['food_medium_2'], data['food_long_2'],
                            data['positive_short_2'], data['positive_medium_2'], data['positive_long_2'], 
                            data['neutral_short_2'], data['neutral_medium_2'], data['neutral_long_2']])
        
        # Save the stacked data using derived_conds
        save_data = {
            'food': food,
            'food_1': food_1,
            'food_2': food_2,

            'positive': positive,
            'positive_1': positive_1,
            'positive_2': positive_2,

            'neutral': neutral,
            'neutral_1': neutral_1,
            'neutral_2': neutral_2,

            'nonfood': nonfood,
            'nonfood_1': nonfood_1,
            'nonfood_2': nonfood_2,

            'salient': salient,
            'salient_1': salient_1,
            'salient_2': salient_2,

            'control': control,
            'control_1': control_1,
            'control_2': control_2,

            'pres_1': pres_1,
            'pres_2': pres_2,
        }

        for filename, data_array in save_data.items():
            save_path = os.path.join(saving_folder, f"{filename}_{suffix}")
            np.save(save_path, data_array)
            print(f"Saved {filename} for subject {file}")
