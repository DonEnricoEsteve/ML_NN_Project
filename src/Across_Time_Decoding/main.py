import mne, glob
import numpy as np 

from mne.decoding import SlidingEstimator, GeneralizingEstimator
from joblib import Parallel, delayed

from config import *
from preprocessing import *
from time_decoding import *
from stats import *
from plots import plot_generalizing, plot_sliding

# Get a list of all `.fif` files in the specified directory
epochs_files = glob.glob(f"{epochs_path}/*.fif")

# Initialize containers for storing decoding results across all subjects
scores_sliding_all, scores_generalization_all = [], []

# Loop over each contrast (e.g., Food vs. Neutral) and its corresponding event codes
for contrast, events_code in contrasts_events_code.items():

    # Loop over each subject and their corresponding epochs file
    for subject, epochs_file in enumerate(epochs_files):

        # Keep track of currently defined variables before subject-level processing begins
        dont_del = set(globals().keys())

        # Load preprocessed epochs
        epochs = mne.read_epochs(epochs_file)

        # Downsample (resample) epochs to the desired sampling rate (200Hz)
        epochs_decim = epochs.copy().resample(sfreq=resampling_rate)

        # Preprocess data: pseudo trials and binary event labels for the contrast
        X_pseudo_all, event_id_bin = preprocess_data(
            epochs=epochs_decim, 
            trials_to_retain=trials_to_retain, 
            contrast_events_code=events_code
        )

        # ---------------- Sliding Estimator ----------------
        # Create the classifier wrapped in a time decoding (sliding) estimator
        time_decod_sliding = stacking_classifier_defenition(time_decoding_classifier=SlidingEstimator)

        # Perform 8 iterations of cross-validated decoding in parallel
        scores_sliding = Parallel(n_jobs=-1)(
            delayed(cross_val_iter)(X=X_pseudo_all, y=event_id_bin, classifier=time_decod_sliding, random_state=i) 
            for i in range(8)
        )

        # Average scores across iterations and append for this subject
        scores_sliding_all.append(np.mean(scores_sliding, axis=0))

        # ---------------- Generalizing Estimator ----------------
        # Create the classifier wrapped in a time generalization estimator
        time_decode_generalization = stacking_classifier_defenition(time_decoding_classifier=GeneralizingEstimator)

        # Perform 8 iterations of time generalization decoding in parallel
        scores_generalization = Parallel(n_jobs=-1)(
            delayed(cross_val_iter)(X=X_pseudo_all, y=event_id_bin, classifier=time_decode_generalization, random_state=i) 
            for i in range(8)
        )

        # Average across iterations and reshape to maintain 3D structure
        scores_sub_gen = np.mean(scores_generalization, axis=0)
        scores_generalization_all.append(np.reshape(scores_sub_gen, (1, scores_sub_gen.shape[0], scores_sub_gen.shape[1])))

        # -------- Memory Management --------
        # Delete all variables that were created during subject-level processing to free memory
        for obj in list(globals().keys()):
            if (not obj.startswith('__') and
                obj not in dont_del) and obj != "dont_del":
                    del globals()[obj]

    # -------- After all subjects processed --------
    # Combine results from all subjects into arrays
    scores_sliding_all = np.vstack(scores_sliding_all)
    scores_generalization_all = np.vstack(scores_generalization_all)

    # Perform statistical testing on the results
    pvalue_list, sem_list = stats_sliding(scores_sliding_all)
    pvalue_mat = stats_generalizing(scores_generalization_all)

    # Save decoding scores and stats to disk
    np.save(file=f"{scores_path}/sliding_estimator_{contrast}_scores.npy", arr=scores_sliding_all)
    np.save(file=f"{scores_path}/generalizing_estimator_{contrast}_scores.npy", arr=scores_generalization_all)
    np.save(file=f"{stats_path}/sliding_estimator_{contrast}_pvalues.npy", arr=np.array(pvalue_list))
    np.save(file=f"{stats_path}/sliding_estimator_{contrast}_sem.npy", arr=np.array(sem_list))
    np.save(file=f"{stats_path}/generalizing_estimator_{contrast}_pvalues.npy", arr=pvalue_mat)

    # -------- Memory Management (again) --------
    # Delete all variables created during contrast-level processing
    for obj in list(globals().keys()):
        if (not obj.startswith('__') and
            obj not in dont_del and obj != "dont_del"):
                del globals()[obj]

    # -------- Plotting --------
    # Generate and save plots for the current contrast
    plot_sliding(contrast)
    plot_generalizing(contrast)
