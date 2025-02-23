from utils import *

def convert_mat_to_dict(file):
    """
    Converts a mat file to a dictionary using the pymatreader module.
    The .mat file for each subject is named datafinalLow_subXXX

    Parameters:
    - file (str) - MEG recordings that underwent signal preprocessing in FieldTrip.  

    Returns: A dictionary with each Matlab field as a key
    """

    try:
        # Use pymatreader functionality to convert .mat to dict
        dict_from_mat = read_mat(file)
        return dict_from_mat
    except PermissionError as e:
        print(f"Permission error in path: {file}. Skipping. \n {e}")
    except Exception as e:
        print(f"A non premission error occured in path: {file}. \n {e}. \n Printing error traceback: \n")
        traceback.print_exc()

def create_mne_info_all(sub_dict):
    """
    Takes information from a dictionary to populate the create_info() function of MNE.

    Parameters:
    - sub_dict (dict) - A dictionary containing information from .mat files  

    Returns: An mne.Info object with information about the sensors and methods of measurement
    """
    try:
        # Take the name of each channel
        ch_names = sub_dict.get("datafinalLow").get("grad").get("label")[0:246]

        # Take only the channels of type MEG ('mag')
        ch_types = np.array(246 * ['mag'])

        # Take the sampling frequency = 1017 Hz
        sfreq = sub_dict.get("datafinalLow").get("fsample")

        # Create the mne.Info object
        mne_info = mne.create_info(ch_names, sfreq, ch_types, verbose=None)
        return mne_info
    
    except Exception as e:
        print(f"an exception has occured: {e}. \n Printing error traceback \n")
        traceback.print_exc()

def convert_dict_to_epochs(sub_dict, mne_info):
    """
    Creates an mne.EpochsArray class that is used in the MNE-Python pipeline

    Parameters:
    - sub_dict (dict) - A dictionary containing information from .mat files
    - mne_info (var) - An mne.Info object with information about the sensors and methods of measurement

    Returns: An mne.EpochsArray class 
    """

    # Extract data and trial info
    data = sub_dict.get("datafinalLow").get("trial") 
    tmin = -0.3
    event_id = np.array(sub_dict.get("datafinalLow").get("trialinfo")[:, 0], dtype=int)
    
    # Identify and remove oddball trials
    oddball_idx = np.where(event_id == 8)[0]  # Ensure you get indices (using [0] to extract the indices array)
    
    # Remove oddball events from the data
    event_id = np.delete(event_id, oddball_idx)
    data = np.delete(data, oddball_idx, axis=0)  # Remove corresponding data rows (trials)
    
    # Create event onset and preceding event arrays
    event_onset = np.arange(len(event_id), dtype=int)
    event_precede = np.zeros(len(event_id), dtype=int)
    
    # Stack the event info into the correct format (onset, preceding, event_id)
    events = np.vstack((event_onset, event_precede, event_id)).T

    print(np.shape(events))  # Should now match the number of trials/data entries (~570, 3)
    print(np.shape(data))    # Ensure that data and events have consistent shapes (~570, 246, 1119)
    
    # Define event_id mapping
    event_mapping = {
        "food/short/rep1": 10, "food/medium/rep1": 12, "food/long/rep1": 14, 
        "food/short/rep2": 20, "food/medium/rep2": 22, "food/long/rep2": 24,
        "positive/short/rep1": 110, "positive/medium/rep1": 112, "positive/long/rep1": 114,
        "positive/short/rep2": 120, "positive/medium/rep2": 122, "positive/long/rep2": 124,
        "neutral/short/rep1": 210, "neutral/medium/rep1": 212, "neutral/long/rep1": 214,
        "neutral/short/rep2": 220, "neutral/medium/rep2": 222, "neutral/long/rep2": 224
    }
    
    # Create the epochs
    baseline = (-0.3, 0)
    epochs = mne.EpochsArray(
        data, mne_info, events=events, tmin=tmin, event_id=event_mapping,
        reject=None, flat=None, reject_tmin=None, reject_tmax=None,
        baseline=baseline, proj=True, on_missing='raise', metadata=None,
        selection=None, drop_log=None, raw_sfreq=None, verbose=None
    )
    
    return epochs

def convert_mat_to_epochsFIF(mat_input_directory: str, fif_output_directory: str):
    """
    Processes the subject data files in the given directory, converts them to MNE Epochs format,
    and saves them as FIF files.

    Parameters:
    - mat_input_directory (str): The path to the directory containing subject data files.
    - fif_output_directory (str): The path to the directory where the processed FIF files will be saved.
    """

    # Define the directory and the file pattern for matching
    directory = Path(mat_input_directory)

    # Initialize the iteration count
    iteration = 0

    # Get a list of all files in the directory that match the pattern. This may change depending on your data.
    file_paths = directory.glob("datafinalLow*")

    # Iterate over the matched files
    for file in file_paths:
        # Take the unique part of the file - subject number XXX
        subject_num = re.split(r"[_.]+", file.name)[1]

        # Convert .mat to dict
        sub_dict = convert_mat_to_dict(file)

        # Create mne.Info object only once as it is same across subjects
        if iteration == 0:
            mne_info = create_mne_info_all(sub_dict)

        # Create mne.EpochsArray class
        epochs = convert_dict_to_epochs(sub_dict, mne_info)

        # Save as FIF files which is the standard format readable in MNE-Python
        save_in_path = os.path.join(fif_output_directory, f"{subject_num}_epo.fif")
        epochs.save(save_in_path, overwrite=True)

        # Change the iteration to prevent the creation of new mne.Info object
        iteration = 1

def split_list(lst, chunk_size):
    """
    Helper function to split a list into chunks of a specified size.
    Used for grouping the conditions based on event ID.

    Parameters:
    - lst (list) - List to be splitted
    - chunk_size (int) - Number of elements in the output

    Returns: A list with n=chunk_size elements
    """

    # Use a list comprehension to create chunks
    # For each index 'i' in the range from 0 to the length of the list with step 'chunk_size'
    # Slice the list from index 'i' to 'i + chunk_size'
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def stack_lags_of_conditions(epochs, conditions):
    """
    Takes the information from each lag duration of each category * repetition and concatenates them.

    Parameters:
    - epochs (var): Instance of Epochs class
    - conditions (list): Information from split_list(). Corresponds to category * repetition

    Returns: A three-dimensional numpy array (np.ndarray)
    """
    
    # The indices 0,1,2 are with respect to how the conditions were originally ordered:
    # e.g. Food_Short_1, Food_Medium_1, Food_Long_1, ...
    short = epochs[conditions[0]].get_data()
    medium = epochs[conditions[1]].get_data()
    long = epochs[conditions[2]].get_data()

    # Vertically stack to create 3D array
    stacked_over_trial = np.vstack((short, medium, long))
    return stacked_over_trial

def convert_epochsFIF_to_npy(fif_input_directory: str, npy_output_directory: str, conds: list, tmin: float = None, tmax: float = None):
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

        # Create a list of 6 conditions (Food_1, Food_2, Positive_1, Positive_2, Neutral_1, Neutral_2)
        groups = split_list(conditions, 3)

        # Save numpy arrays for each condition group
        for group, cond in zip(groups, conds):
            concat_condition_data = stack_lags_of_conditions(epochs_toi, group)
            path = os.path.join(saving_folder, cond)
            np.save(path, concat_condition_data)

        print(f"Processed and saved data for {file.name}.")

def halve_trials(data):
    """
    Averages every two consecutive trials along the first axis (trial axis).
    If the number of trials is odd, averages the last trial with the previous ones.
    
    Parameters:
    data (numpy.ndarray): 3D array of shape (n_trials, n_channels, n_time)
    
    Returns:
    numpy.ndarray: 3D array with averaged trials, shape (n_trials // 2, n_channels, n_time) for even trials,
                    and (n_trials // 2 + 1, n_channels, n_time) for odd trials.
    """
    n_trials, n_channels, n_time = data.shape
    
    if n_trials % 2 == 0:
        # For even number of trials, average every two trials
        averaged_data = data.reshape(n_trials // 2, 2, n_channels, n_time).mean(axis=1)
    else:
        # For odd number of trials, average every two trials except the last
        # Reshape and average the pairs of trials
        paired_data = data[:-1].reshape(n_trials // 2, 2, n_channels, n_time).mean(axis=1)
        
        # Average the last two trials
        last_trial = data[-2:].mean(axis=0)  # Average the last two trials
        
        # Concatenate the averaged pairs with the averaged last trial
        averaged_data = np.vstack([paired_data, last_trial[np.newaxis]])
    
    return averaged_data

def derive_class_labels(npy_IO_directory):
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
        condition_files = {
            'food_1': os.path.join(subject_dir, 'food_1.npy'),
            'positive_1': os.path.join(subject_dir, 'positive_1.npy'),
            'neutral_1': os.path.join(subject_dir, 'neutral_1.npy'),
            'food_2': os.path.join(subject_dir, 'food_2.npy'),
            'positive_2': os.path.join(subject_dir, 'positive_2.npy'),
            'neutral_2': os.path.join(subject_dir, 'neutral_2.npy')
        }

        # Check if all files exist
        if all(os.path.exists(path) for path in condition_files.values()):
            # Load all condition files
            data = {key: np.load(path) for key, path in condition_files.items()}
            print(f"Loaded data for subject {file}")
        else:
            print(f"Missing .npy files for subject {file}, skipping...")
            continue

        # Stack the data for various conditions
        food = np.vstack([data['food_1'], data['food_2']])
        nonfood = np.vstack([data['positive_1'], data['neutral_1'], data['positive_2'], data['neutral_2']])
        nonfood_1 = np.vstack([data['positive_1'], data['neutral_1']])
        nonfood_2 = np.vstack([data['positive_2'], data['neutral_2']])
        positive = np.vstack([data['positive_1'], data['positive_2']])
        neutral = np.vstack([data['neutral_1'], data['neutral_2']])
        pres_1 = np.vstack([data['food_1'], data['positive_1'], data['neutral_1']])
        pres_2 = np.vstack([data['food_2'], data['positive_2'], data['neutral_2']])

        # Reduce non-food trials by averaging every two trials (to remove class imbalance)
        nonfood_eq = halve_trials(nonfood)
        nonfood_1_eq = halve_trials(nonfood_1)
        nonfood_2_eq = halve_trials(nonfood_2)

        # Save the stacked data using derived_conds
        save_data = {
            'food.npy': food,
            'nonfood.npy': nonfood,
            'nonfood_1.npy': nonfood_1,
            'nonfood_2.npy': nonfood_2,
            'nonfood_eq.npy': nonfood_eq,
            'nonfood_1_eq.npy': nonfood_1_eq,
            'nonfood_2_eq.npy': nonfood_2_eq,
            'positive.npy': positive,
            'neutral.npy': neutral,
            'pres_1.npy': pres_1,
            'pres_2.npy': pres_2
        }

        for filename, data_array in save_data.items():
            if filename in derived_conds:  # Only save the derived conditions
                save_path = os.path.join(saving_folder, filename)
                np.save(save_path, data_array)
                print(f"Saved {filename} for subject {file}")