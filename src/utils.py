# ============================================== #
# Import necessary libraries
# ============================================== #

from pymatreader import read_mat
from pathlib import Path
import numpy as np
import os
import glob
import mne
import os   
import traceback
import re
from joblib import Parallel, delayed
from matplotlib.ticker import AutoMinorLocator
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import StackingClassifier
from sklearn.utils import shuffle
from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.manifold import MDS


# ============================================== #
# Part 0: Initialization
# ============================================== #

# Define seed for reproducibility
np.random.seed(42)

# Set working directory. Set your own directory.
wd = "/Volumes/Seagate/ML_Project/MARCH_ANALYSIS"

# Set directory where preprocessed MEG files (.fif) are (to be) located
fif_directory = "/Volumes/Seagate/ML_Project/Evoked_fif"


# ============================================== #
# Part 1: Data preparation and preprocessing
# ============================================== #

# Set directory where preprocessed MEG files (.npy) are (to be) located
# Also used in plotting Figure S2
npys = [f"{wd}/BL_orig", f"{wd}/PS_orig", f"{wd}/M100_orig", f"{wd}/M200_orig", f"{wd}/M300_orig", f"{wd}/MLPP_orig"]

# Define event-related field (ERF) components
component_start_times = [-0.299, 0, 0.118, 0.171, 0.239, 0.350]
component_end_times = [0, 0.799, 0.155, 0.217, 0.332, 0.799]
component_labels = ["BL", "PS", "M100", "M200", "M300", "MLPP"] # Also used during RSA

# Define the folder names where the converted numpy arrays for each time window are (to be) located
ppc_names = ["BL_Evoked_PPC", "PS_Evoked_PPC", "M100_Evoked_PPC", "M200_Evoked_PPC", "M300_Evoked_PPC", "MLPP_Evoked_PPC"]

# Define the number of groups (controls sample size) to be used for pseudo-trial calculation
grps = [5, 10]

# Define the folder names where the preprocessed numpy arrays for each relative sample size are (to be) located
ppc_folders = [f"{wd}/5PT", f"{wd}/10PT"]

# Define a suffix to be added for the preprocessed numpy arrays to differentiate these from the raw arrays
sf = "dec_pseudo_PCA"


# ============================================== #
# Part 2: General decoding
# ============================================== #

# Define general directory for results
results_dir = f"{wd}/Results_03_17"

# Define the folder directory containing the window- and sample size-specific preprocessed numpy arrays 
ppc_subfolders = [f"{wd}/5PT/BL_Evoked_PPC", f"{wd}/10PT/BL_Evoked_PPC", f"{wd}/5PT/BL_Evoked_PPC", f"{wd}/10PT/BL_Evoked_PPC",
                  f"{wd}/5PT/M100_Evoked_PPC", f"{wd}/10PT/M100_Evoked_PPC", f"{wd}/5PT/M100_Evoked_PPC", f"{wd}/10PT/M100_Evoked_PPC",
                  f"{wd}/5PT/M200_Evoked_PPC", f"{wd}/10PT/M200_Evoked_PPC", f"{wd}/5PT/M200_Evoked_PPC", f"{wd}/10PT/M200_Evoked_PPC",
                  f"{wd}/5PT/M300_Evoked_PPC", f"{wd}/10PT/M300_Evoked_PPC", f"{wd}/5PT/M300_Evoked_PPC", f"{wd}/10PT/M300_Evoked_PPC",
                  f"{wd}/5PT/MLPP_Evoked_PPC", f"{wd}/10PT/MLPP_Evoked_PPC", f"{wd}/5PT/MLPP_Evoked_PPC", f"{wd}/10PT/MLPP_Evoked_PPC",
                  f"{wd}/5PT/PS_Evoked_PPC", f"{wd}/10PT/PS_Evoked_PPC", f"{wd}/5PT/PS_Evoked_PPC", f"{wd}/10PT/PS_Evoked_PPC"]

# Define the percentage of explained variance (controls feature size) to be used for PCA
expl_var_percs = [0.90, 0.90, 0.70, 0.70,
                  0.90, 0.90, 0.70, 0.70,
                  0.90, 0.90, 0.70, 0.70,
                  0.90, 0.90, 0.70, 0.70,
                  0.90, 0.90, 0.70, 0.70,
                  0.90, 0.90, 0.70, 0.70]

# Define suffixes for output filenames (numpy vector of length 42 containing the within-subject decoding accuracies) 
# Also involved in plotting results
fnames = ["BL_5_90", "BL_10_90", "BL_5_70", "BL_10_70",
          "M100_5_90", "M100_10_90", "M100_5_70", "M100_10_70",
          "M200_5_90", "M200_10_90", "M200_5_70", "M200_10_70",
          "M300_5_90", "M300_10_90", "M300_5_70", "M300_10_70",
          "MLPP_5_90", "MLPP_10_90", "MLPP_5_70", "MLPP_10_70",
          "PS_5_90", "PS_10_90", "PS_5_70", "PS_10_70"]

# Define the binary tasks to be performed (n=25)
binary_tasks = {
    # NEUTRAL VS SALIENT
    "TS": [f"neutral_{sf}", f"salient_{sf}"],
    "TS1": [f"neutral_1_{sf}", f"salient_1_{sf}"],
    "TS2": [f"neutral_2_{sf}", f"salient_2_{sf}"],

    # FOOD VS NONFOOD
    "FN": [f"food_{sf}", f"nonfood_{sf}"],
    "FN1": [f"food_1_{sf}", f"nonfood_1_{sf}"],
    "FN2": [f"food_2_{sf}", f"nonfood_2_{sf}"],

    # POSITIVE VS CONTROL
    "PC": [f"positive_{sf}", f"control_{sf}"],
    "PC1": [f"positive_1_{sf}", f"control_1_{sf}"],
    "PC2": [f"positive_2_{sf}", f"control_2_{sf}"],

    # SOLO
    "F12": [f"food_1_{sf}", f"food_2_{sf}"],
    "P12": [f"positive_1_{sf}", f"positive_2_{sf}"],
    "T12": [f"neutral_1_{sf}", f"neutral_2_{sf}"],

    # SALIENT-RELATED (FOOD AND POSITIVE)
    "FP": [f"food_{sf}", f"positive_{sf}"],
    "FP1": [f"food_1_{sf}", f"positive_1_{sf}"],
    "FP2": [f"food_2_{sf}", f"positive_2_{sf}"],
    "S12": [f"salient_1_{sf}", f"salient_2_{sf}"],

    # NONFOOD-RELATED (POSITIVE AND NEUTRAL)
    "PT": [f"positive_{sf}", f"neutral_{sf}"],
    "PT1": [f"positive_1_{sf}", f"neutral_1_{sf}"],
    "PT2": [f"positive_2_{sf}", f"neutral_2_{sf}"],
    "N12": [f"nonfood_1_{sf}", f"nonfood_2_{sf}"],

    # CONTROL-RELATED (FOOD AND NEUTRAL)
    "FT": [f"food_{sf}", f"neutral_{sf}"],
    "FT1": [f"food_1_{sf}", f"neutral_1_{sf}"],
    "FT2": [f"food_2_{sf}", f"neutral_2_{sf}"],
    "C12": [f"control_1_{sf}", f"control_2_{sf}"],

    "12": [f"pres_1_{sf}", f"pres_2_{sf}"],
}

# Define the multi-class classification tasks to be performed (n=10)
multiclass_tasks = {
    # 4-class using the derived categories
    "4-FN-12": [f"food_1_{sf}", f"food_2_{sf}", f"nonfood_1_{sf}", f"nonfood_2_{sf}"],
    "4-TS-12": [f"neutral_1_{sf}", f"neutral_2_{sf}", f"salient_1_{sf}", f"salient_2_{sf}"],
    "4-PC-12": [f"positive_1_{sf}", f"positive_2_{sf}", f"control_1_{sf}", f"control_2_{sf}"],

    # 3-class
    "3-FPT": [f"food_{sf}", f"positive_{sf}", f"neutral_{sf}"],
    "3-FPT-1": [f"food_1_{sf}", f"positive_1_{sf}", f"neutral_1_{sf}"],
    "3-FPT-2": [f"food_2_{sf}", f"positive_2_{sf}", f"neutral_2_{sf}"],
    
    # 4-class using the original categories
    "4-FP-12": [f"food_1_{sf}", f"food_2_{sf}", f"positive_1_{sf}", f"positive_2_{sf}"],
    "4-FT-12": [f"food_1_{sf}", f"food_2_{sf}", f"neutral_1_{sf}", f"neutral_2_{sf}"],
    "4-PT-12": [f"positive_1_{sf}", f"positive_2_{sf}", f"neutral_1_{sf}", f"neutral_2_{sf}"],

    # 6-class
    "6-FPT-12": [f"food_1_{sf}", f"food_2_{sf}", f"positive_1_{sf}", 
                 f"positive_2_{sf}", f"neutral_1_{sf}", f"neutral_2_{sf}"],
}

# Define the chance levels for the multi-class classification tasks for one-tailed Wilcoxon signed-rank test
# Binary tasks all have a chance of 0.50
multi_chance = {
    "3-FPT": 0.333,
    "3-FPT-1": 0.333,
    "3-FPT-2": 0.333,

    "4-FP-12": 0.250,
    "4-FT-12": 0.250,
    "4-PT-12": 0.250,

    "4-FN-12": 0.250,
    "4-TS-12": 0.250,
    "4-PC-12": 0.250,

    "6-FPT-12": 0.166
}

# ============================================================ #
# Part 3: Plot summary statistics and general decoding results
# ============================================================ #

# Set directory where summary statistics are (to be) located
SS_dir = f"{wd}/Summary_statistics_plots"

# Define the filenames of each output from general decoding (n=24) for plotting Figure 3
binary_files_for_Fig3 = [
    "BL_5_70_accuracies_binary.npy", "BL_5_90_accuracies_binary.npy", "BL_10_70_accuracies_binary.npy", "BL_10_90_accuracies_binary.npy",
    "PS_5_70_accuracies_binary.npy", "PS_5_90_accuracies_binary.npy", "PS_10_70_accuracies_binary.npy", "PS_10_90_accuracies_binary.npy",
    "m100_5_70_accuracies_binary.npy", "m100_5_90_accuracies_binary.npy", "m100_10_70_accuracies_binary.npy", "m100_10_90_accuracies_binary.npy",
    "m200_5_70_accuracies_binary.npy", "m200_5_90_accuracies_binary.npy", "m200_10_70_accuracies_binary.npy", "m200_10_90_accuracies_binary.npy",
    "m300_5_70_accuracies_binary.npy", "m300_5_90_accuracies_binary.npy", "m300_10_70_accuracies_binary.npy", "m300_10_90_accuracies_binary.npy",
    "mLPP_5_70_accuracies_binary.npy", "mLPP_5_90_accuracies_binary.npy", "mLPP_10_70_accuracies_binary.npy", "mLPP_10_90_accuracies_binary.npy"
]

# Load the final_rms over the entire trial time window for plotting Figure S1
all_final_rms = np.load("/Volumes/Seagate/ML_Project/MARCH_ANALYSIS/Summary_statistics_plots/ALL_rms.npy", allow_pickle=True)

# Define condition groups for plotting Figure S1
SS_conditions = ['food_short_rep1', 'food_medium_rep1', 'food_long_rep1', 
                  'food_short_rep2', 'food_medium_rep2', 'food_long_rep2',
                  'positive_short_rep1', 'positive_medium_rep1', 'positive_long_rep1',
                  'positive_short_rep2', 'positive_medium_rep2', 'positive_long_rep2',
                  'neutral_short_rep1', 'neutral_medium_rep1', 'neutral_long_rep1',
                  'neutral_short_rep2', 'neutral_medium_rep2', 'neutral_long_rep2'
                  ]

SS_labels = ["Food-Short-1st", "Food-Medium-1st", "Food-Long-1st",
             "Food-Short-2nd", "Food-Medium-2nd", "Food-Long-2nd",
             "Positive-Short-1st", "Positive-Medium-1st", "Positive-Long-1st",
             "Positive-Short-2nd", "Positive-Medium-2nd", "Positive-Long-2nd",
             "Neutral-Short-1st", "Neutral-Medium-1st", "Neutral-Long-1st",
             "Neutral-Short-2nd", "Neutral-Medium-2nd", "Neutral-Long-2nd"
             ]

condition_groups = [
    (SS_labels[0:6], SS_conditions[0:6], "(a)", "Food"),
    (SS_labels[6:12], SS_conditions[6:12], "(b)", "Positive"),
    (SS_labels[12:18], SS_conditions[12:18], "(c)", "Neutral")
]

# Define filenames for for RMS and SEM calculation for the original conditions during each time window for plotting Figure S2
rms_names = ["BL_rms", "PS_rms", "M100_rms", "M200_rms", "M300_rms", "MLPP_rms"]
sem_names = ["BL_sem", "PS_sem", "M100_sem", "M200_sem", "M300_sem", "MLPP_sem"]
times = [300, 800, 39, 46, 93, 449] # Number of time points during each window

# Define labels and colors for plotting Figure S2
components_labels = [
    ("(a)", "Baseline Window", "black"),
    ("(b)", "Post-Stimulus Window", "lightgray"),
    ("(c)", "M100 Window", "lightskyblue"),
    ("(d)", "M200 Window", "lightgreen"),
    ("(e)", "M300 Window", "salmon"),
    ("(f)", "MLPP Window", "#D1A6D0")
]


# ============================================================ #
# Part 5: Representational Similarity Analysis (RSA)
# ============================================================ #

# Set directory to save RSA files
RSA_wd = f"{wd}/RSA_Results_03_27"

# Define folder directories for performing RSA (only 10-90%)
rsa_ppcs = [f"{wd}/10PT/BL_Evoked_PPC", f"{wd}/10PT/M100_Evoked_PPC", f"{wd}/10PT/M200_Evoked_PPC", 
            f"{wd}/10PT/M300_Evoked_PPC", f"{wd}/10PT/MLPP_Evoked_PPC", f"{wd}/10PT/PS_Evoked_PPC"]

# Define file names for each ORIGINAL condition (n=18, category X lag X presentation) after PCA
pca_orig_cond_list = [f"food_short_rep1_{sf}", "food_medium_rep1_dec_pseudo_PCA", "food_long_rep1_dec_pseudo_PCA",
             "food_short_rep2_dec_pseudo_PCA", "food_medium_rep2_dec_pseudo_PCA", "food_long_rep2_dec_pseudo_PCA",
             "positive_short_rep1_dec_pseudo_PCA", "positive_medium_rep1_dec_pseudo_PCA", "positive_long_rep1_dec_pseudo_PCA",
             "positive_short_rep2_dec_pseudo_PCA", "positive_medium_rep2_dec_pseudo_PCA", "positive_long_rep2_dec_pseudo_PCA",
             "neutral_short_rep1_dec_pseudo_PCA", "neutral_medium_rep1_dec_pseudo_PCA", "neutral_long_rep1_dec_pseudo_PCA",
             "neutral_short_rep2_dec_pseudo_PCA", "neutral_medium_rep2_dec_pseudo_PCA", "neutral_long_rep2_dec_pseudo_PCA",
             ]

# Define shorthand labels for each condition used during plotting
cond_short_labels = ["FS1", "FM1", "FL1", "FS2", "FM2", "FL2", "PS1", "PM1", "PL1", 
                     "PS2", "PM2", "PL2", "NS1", "NM1", "NL1", "NS2", "NM2", "NL2"]
