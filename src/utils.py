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
wd = "Z:\Don\ML_WD" 

# Define event-related field (ERF) components
component_start_times = [0, 0.118, 0.171, 0.239, 0.350]
component_end_times = [0.799, 0.155, 0.217, 0.332, 0.799]
component_labels = ['PS', 'm100', 'm200', 'm300', 'mLPP']

components_with_colors = [
    (component_start_times[0], component_end_times[0], "(a)", "Post-Stimulus Window (0-800 ms)", "lightgray"),
    (component_start_times[1], component_end_times[1], "(b)", "m100 Component (118-155 ms)", "lightskyblue"),
    (component_start_times[2], component_end_times[2], "(c)", "m200 Component (171-217 ms)", "lightgreen"),
    (component_start_times[3], component_end_times[3], "(d)", "m300 Component (239-332 ms)", "salmon"),
    (component_start_times[4], component_end_times[4], "(e)", "mLPP Component (350-800 ms)", "#D1A6D0")
]

# ============================================== #
# Part 1: Convert MNE epochs to Numpy arrays
# ============================================== #

# Set directory where preprocessed MEG files (.mat) are located
mat_directory = f"{wd}/Evoked_mat"

# Set directory where preprocessed MEG files (.fif) are (to be) located
fif_directory = f"{wd}/Evoked_fif"

# Set directory where preprocessed MEG files (.npy) are (to be) located
npy_directory = f"{wd}/Evoked_npy"

# Define the names of the output numpy arrays
base_conds = ['food_1.npy', 'food_2.npy', 'positive_1.npy', 'positive_2.npy', 'neutral_1.npy', 'neutral_2.npy']

# Define the names of the output numpy arrays derived from the base conditions
derived_conds = ['food.npy', 'nonfood.npy', 'nonfood_1.npy', 'nonfood_2.npy', 
                 'positive.npy', 'neutral.npy', 'pres_1.npy', 'pres_2.npy']

# ============================================== #
# Part 2: Data Preprocessing
# ============================================== #

# Set directory where temporally decimated arrays are (to be) located
decimated_npy_directory = f"{wd}/Evoked_dec"

# Set directory where calculated pseudo-trial arrays are (to be) located
pseudotrial_npy_directory = f"{wd}/Evoked_pseudo"

# Set directory where PCA arrays are (to be) located
PCA_npy_directory = f"{wd}/Evoked_PCA"

# ============================================== #
# Part 3: General decoding
# ============================================== #

# Define general directory for results
results_dir = f"{wd}/Results"

# Define the binary classification tasks to be performed
# You may add your own here
binary_tasks = {
    "12": ["pres_1_dec_pseudo_PCA", "pres_2_dec_pseudo_PCA"],
    "FN": ["food_dec_pseudo_PCA", "nonfood_dec_pseudo_PCA"],
    "FNeq": ["food_dec_pseudo_PCA", "nonfood_eq_dec_pseudo_PCA"],
    "FP": ["food_dec_pseudo_PCA", "positive_dec_pseudo_PCA"],
    "FT": ["food_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "PT": ["positive_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "FN1": ["food_1_dec_pseudo_PCA", "nonfood_1_dec_pseudo_PCA"],
    "FN1eq": ["food_dec_pseudo_PCA", "nonfood_1_eq_dec_pseudo_PCA"],
    "FN2": ["food_2_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],
    "FN2eq": ["food_dec_pseudo_PCA", "nonfood_2_eq_dec_pseudo_PCA"],
    "F12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA"],
    "P12": ["positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA"],
    "T12": ["neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "N12": ["nonfood_1_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],
    "N12eq": ["nonfood_1_eq_dec_pseudo_PCA", "nonfood_2_eq_dec_pseudo_PCA"],
}

# Define the multi-class classification tasks to be performed
# You may add your own here
multiclass_tasks = {
    "3A": ["food_dec_pseudo_PCA", "positive_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "3B": ["food_1_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA"],
    "3C": ["food_2_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "4A": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "nonfood_1_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],
    "4Aeq": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "nonfood_1_eq_dec_pseudo_PCA", "nonfood_2_eq_dec_pseudo_PCA"],
    "4B": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA"],
    "4C": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "6": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA",
          "neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"]
}

# Define the chance levels for the binary classification tasks for Wilcoxon signed-rank test
binary_chance = {
    "12": 0.50, "FN": 0.66, "FNeq": 0.50, "FP": 0.50, "FT": 0.50, "PT": 0.50, "FN1": 0.66, "FN1eq": 0.50, 
    "FN2": 0.66, "FN2eq": 0.50, "F12": 0.50, "P12": 0.50, "T12": 0.50, "N12": 0.50, "N12eq": 0.50
}

# Define the chance levels for the multi-class classification tasks for Wilcoxon signed-rank test
multi_chance = {
    "3A": 0.33, "3B": 0.33, "3C": 0.33, 
    "4A": 0.33, "4Aeq": 0.25, "4B": 0.25, 
    "4C": 0.25, "6": 0.166
}

# ============================================================ #
# Part 4: Plot summary statistics and general decoding results
# ============================================================ #

# Define the filenames of each window for plotting Figure 2
binary_files_for_Fig2 = [
    "list_binary_accuracies_40_10_10.npy",        # Post-stimulus
    "list_binary_accuracies_M100_2_10_13.npy",
    "list_binary_accuracies_M200_2_10_10.npy", 
    "list_binary_accuracies_M300_4_10_10.npy", 
    "list_binary_accuracies_LPP_20_10_11.npy"
]

# Set directory where summary statistics are (to be) located
SS_dir = f"{wd}/Summary_statistics_plots"

# Set all condition names without .npy extension
SS_conditions = ['food', 'positive', 'neutral', 'nonfood', 'food_1', 'food_2', 'positive_1', 'positive_2', 
              'neutral_1', 'neutral_2', 'nonfood_1', 'nonfood_2', 'pres_1', 'pres_2']

# Define labels formally for each condition
SS_labels = ["Food", "Positive", "Neutral", "Nonfood", 'Food 1', "Food 2", "Positive 1", "Positive 2", 
          "Neutral 1", "Neutral 2", "Nonfood 1", "Nonfood 2", "Presentation 1", "Presentation 2"]

# Define condition groups
condition_groups = [
    (SS_conditions[0:4], "(a)", "Category"),
    (SS_conditions[12:14], "(b)", "Presentation"),
    (SS_conditions[4:12], "(c)", "Category and Presentation")
]

# Define the directory containing the arrays before preprocessing during the post-stimulus window for plotting Figure S1
input_dir_for_FigS1 = f'{wd}/Evoked_npy'

# ============================================================ #
# Part 7: Representational Similarity Analysis (RSA)
# ============================================================ #

# Define file names for each ORIGINAL condition (n=18, category X lag X presentation)
orig_cond_list = ['food_short_1.npy', 'food_medium_1.npy', 'food_long_1.npy', 
                  'food_short_2.npy', 'food_medium_2.npy', 'food_long_2.npy',
                  'positive_short_1.npy', 'positive_medium_1.npy', 'positive_long_1.npy',
                  'positive_short_2.npy', 'positive_medium_2.npy', 'positive_long_2.npy',
                  'neutral_short_1.npy', 'neutral_medium_1.npy', 'neutral_long_1.npy',
                  'neutral_short_2.npy', 'neutral_medium_2.npy', 'neutral_long_2.npy']

# Define file names for each ORIGINAL condition (n=18, category X lag X presentation) after PCA
pca_orig_cond_list = ['food_short_rep1_pca', 'food_medium_rep1_pca', 'food_long_rep1_pca', 
                 'food_short_rep2_pca', 'food_medium_rep2_pca', 'food_long_rep2_pca',
                 'positive_short_rep1_pca', 'positive_medium_rep1_pca', 'positive_long_rep1_pca', 
                 'positive_short_rep2_pca', 'positive_medium_rep2_pca', 'positive_long_rep2_pca',
                 'neutral_short_rep1_pca', 'neutral_medium_rep1_pca', 'neutral_long_rep1_pca', 
                 'neutral_short_rep2_pca', 'neutral_medium_rep2_pca', 'neutral_long_rep2_pca']

# Define shorthand labels for each condition
cond_short_labels = ["FS1", "FM1", "FL1", "FS2", "FM2", "FL2", "PS1", "PM1", "PL1", 
                     "PS2", "PM2", "PL2", "NS1", "NM1", "NL1", "NS2", "NM2", "NL2"]

# Set directory to save RSA files
RSA_wd = f"{wd}/RSA_files"