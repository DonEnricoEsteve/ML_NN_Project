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
wd = "Z:\Don\ML_Project\MARCH_ANALYSIS"

# Define event-related field (ERF) components (ADD BL)
component_start_times = [-0.3, 0, 0.118, 0.171, 0.239, 0.350]
component_end_times = [0, 0.799, 0.155, 0.217, 0.332, 0.799]

# ============================================== #
# Part 1: Convert MNE epochs to Numpy arrays
# ============================================== #

# Set directory where preprocessed MEG files (.fif) are (to be) located
fif_directory = "Z:\Don\ML_Project\Evoked_fif"

# Set directory where preprocessed MEG files (.npy) are (to be) located
npy_directory = f"{wd}/NEW_npys/M200_orig"

PPC_directory = f"{wd}/NEW_npys/10PT/M200_Evoked_PPC"

SF = "dec_pseudo_PCA"

# Define the names of the output numpy arrays derived from the base conditions
derived_conds = [
    'food.npy', 'food_1.npy', 'food_2.npy', 
    'positive.npy', 'positive_1.npy', 'positive_2.npy',
    'neutral.npy', 'neutral_1.npy', 'neutral_2.npy',  
    'nonfood.npy', 'nonfood_1.npy', 'nonfood_2.npy',
    'salient.npy', 'salient_1.npy', 'salient_2.npy',
    'control.npy','control_1.npy','control_2.npy',
    'pres_1.npy', 'pres_2.npy']

# ============================================== #
# Part 2: Data Preprocessing
# ============================================== #

# # Set directory where temporally decimated arrays are (to be) located
# decimated_npy_directory = f"{wd}/Evoked_dec"

# # Set directory where calculated pseudo-trial arrays are (to be) located
# pseudotrial_npy_directory = f"{wd}/Evoked_pseudo"

# # Set directory where PCA arrays are (to be) located
# PCA_npy_directory = f"{wd}/Evoked_PCA"

# ============================================== #
# Part 3: General decoding
# ============================================== #

# Define general directory for results
results_dir = f"{wd}/Results_03_17"

# Define the binary classification tasks to be performed
# FOLLOWING THE FRAMEWORK GROUPING
binary_tasks = {
    # ALL-RELATED
    # NEUTRAL VS SALIENT
    "TS": ["neutral_dec_pseudo_PCA", "salient_dec_pseudo_PCA"],
    "TS1": ["neutral_1_dec_pseudo_PCA", "salient_1_dec_pseudo_PCA"],
    "TS2": ["neutral_2_dec_pseudo_PCA", "salient_2_dec_pseudo_PCA"],

    # FOOD VS NONFOOD
    "FN": ["food_dec_pseudo_PCA", "nonfood_dec_pseudo_PCA"],
    "FN1": ["food_1_dec_pseudo_PCA", "nonfood_1_dec_pseudo_PCA"],
    "FN2": ["food_2_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],

    # POSITIVE VS CONTROL
    "PC": ["positive_dec_pseudo_PCA", "control_dec_pseudo_PCA"],
    "PC1": ["positive_1_dec_pseudo_PCA", "control_1_dec_pseudo_PCA"],
    "PC2": ["positive_2_dec_pseudo_PCA", "control_2_dec_pseudo_PCA"],

    # SOLO
    "F12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA"],
    "P12": ["positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA"],
    "T12": ["neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],

    # SALIENT-RELATED (FOOD AND POSITIVE)
    "FP": ["food_dec_pseudo_PCA", "positive_dec_pseudo_PCA"],
    "FP1": ["food_1_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA"],
    "FP2": ["food_2_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA"],
    "S12": ["salient_1_dec_pseudo_PCA", "salient_2_dec_pseudo_PCA"],

    # NONFOOD-RELATED (POSITIVE AND NEUTRAL)
    "PT": ["positive_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "PT1": ["positive_1_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA"],
    "PT2": ["positive_2_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "N12": ["nonfood_1_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],

    # CONTROL-RELATED (FOOD AND NEUTRAL)
    "FT": ["food_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "FT1": ["food_1_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA"],
    "FT2": ["food_2_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "C12": ["control_1_dec_pseudo_PCA", "control_2_dec_pseudo_PCA"],

    "12": ["pres_1_dec_pseudo_PCA", "pres_2_dec_pseudo_PCA"],
}

# Define the multi-class classification tasks to be performed
# By relative difficulty
multiclass_tasks = {
    "4-FN-12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "nonfood_1_dec_pseudo_PCA", "nonfood_2_dec_pseudo_PCA"],
    "4-TS-12": ["neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA", "salient_1_dec_pseudo_PCA", "salient_2_dec_pseudo_PCA"],
    "4-PC-12": ["positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA", "control_1_dec_pseudo_PCA", "control_2_dec_pseudo_PCA"],

    "3-FPT": ["food_dec_pseudo_PCA", "positive_dec_pseudo_PCA", "neutral_dec_pseudo_PCA"],
    "3-FPT-1": ["food_1_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA"],
    "3-FPT-2": ["food_2_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    
    "4-FP-12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA"],
    "4-FT-12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
    "4-PT-12": ["positive_1_dec_pseudo_PCA", "positive_2_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],

    "6-FPT-12": ["food_1_dec_pseudo_PCA", "food_2_dec_pseudo_PCA", "positive_1_dec_pseudo_PCA", 
                 "positive_2_dec_pseudo_PCA", "neutral_1_dec_pseudo_PCA", "neutral_2_dec_pseudo_PCA"],
}

# Define the chance levels for the binary classification tasks for Wilcoxon signed-rank test
binary_chance = {
    # SOLO
    "F12": 0.500,
    "P12": 0.500,
    "T12": 0.500,

    # SALIENT-RELATED (FOOD AND POSITIVE)
    "FP": 0.500,
    "FP1": 0.500,
    "FP2": 0.500,
    "S12": 0.500,

    # NONFOOD-RELATED (POSITIVE AND NEUTRAL)
    "PT": 0.500,
    "PT1": 0.500,
    "PT2": 0.500,
    "N12": 0.500,

    # CONTROL-RELATED (FOOD AND NEUTRAL)
    "FT": 0.500,
    "FT1": 0.500,
    "FT2": 0.500,
    "C12": 0.500,

    # ALL-RELATED
    # NEUTRAL VS SALIENT
    "TS": 0.500,
    "TS1": 0.500,
    "TS2": 0.500,

    # FOOD VS NONFOOD
    "FN": 0.500,
    "FN1": 0.500,
    "FN2": 0.500,

    # POSITIVE VS CONTROL
    "PC": 0.500,
    "PC1": 0.500,
    "PC2": 0.500,

    "12": 0.500,
}

# Define the chance levels for the multi-class classification tasks for Wilcoxon signed-rank test
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
# Part 4: Plot summary statistics and general decoding results
# ============================================================ #

# # Define the filenames of each window for plotting Figure 2
# binary_files_for_Fig2 = [
#     "list_binary_accuracies_40_10_10.npy",        # Post-stimulus
#     "list_binary_accuracies_M100_2_10_13.npy",
#     "list_binary_accuracies_M200_2_10_10.npy", 
#     "list_binary_accuracies_M300_4_10_10.npy", 
#     "list_binary_accuracies_LPP_20_10_11.npy"
# ]

# # Set directory where summary statistics are (to be) located
# SS_dir = f"{wd}/Summary_statistics_plots"

# # Set all condition names without .npy extension
# SS_conditions = ['food', 'positive', 'neutral', 'nonfood', 'food_1', 'food_2', 'positive_1', 'positive_2', 
#               'neutral_1', 'neutral_2', 'nonfood_1', 'nonfood_2', 'pres_1', 'pres_2']

# # Define labels formally for each condition
# SS_labels = ["Food", "Positive", "Neutral", "Nonfood", 'Food 1', "Food 2", "Positive 1", "Positive 2", 
#           "Neutral 1", "Neutral 2", "Nonfood 1", "Nonfood 2", "Presentation 1", "Presentation 2"]

# # Define condition groups
# condition_groups = [
#     (SS_conditions[0:4], "(a)", "Category"),
#     (SS_conditions[12:14], "(b)", "Presentation"),
#     (SS_conditions[4:12], "(c)", "Category and Presentation")
# ]

# # Define the directory containing the arrays before preprocessing during the post-stimulus window for plotting Figure S1
# input_dir_for_FigS1 = f'{wd}/Evoked_npy'

# ============================================================ #
# Part 7: Representational Similarity Analysis (RSA)
# ============================================================ #

# # Define file names for each ORIGINAL condition (n=18, category X lag X presentation)
# orig_cond_list = ['food_short_1.npy', 'food_medium_1.npy', 'food_long_1.npy', 
#                   'food_short_2.npy', 'food_medium_2.npy', 'food_long_2.npy',
#                   'positive_short_1.npy', 'positive_medium_1.npy', 'positive_long_1.npy',
#                   'positive_short_2.npy', 'positive_medium_2.npy', 'positive_long_2.npy',
#                   'neutral_short_1.npy', 'neutral_medium_1.npy', 'neutral_long_1.npy',
#                   'neutral_short_2.npy', 'neutral_medium_2.npy', 'neutral_long_2.npy']

# # Define file names for each ORIGINAL condition (n=18, category X lag X presentation) after PCA
# pca_orig_cond_list = ['food_short_rep1_pca', 'food_medium_rep1_pca', 'food_long_rep1_pca', 
#                  'food_short_rep2_pca', 'food_medium_rep2_pca', 'food_long_rep2_pca',
#                  'positive_short_rep1_pca', 'positive_medium_rep1_pca', 'positive_long_rep1_pca', 
#                  'positive_short_rep2_pca', 'positive_medium_rep2_pca', 'positive_long_rep2_pca',
#                  'neutral_short_rep1_pca', 'neutral_medium_rep1_pca', 'neutral_long_rep1_pca', 
#                  'neutral_short_rep2_pca', 'neutral_medium_rep2_pca', 'neutral_long_rep2_pca']

# # Define shorthand labels for each condition
# cond_short_labels = ["FS1", "FM1", "FL1", "FS2", "FM2", "FL2", "PS1", "PM1", "PL1", 
#                      "PS2", "PM2", "PL2", "NS1", "NM1", "NL1", "NS2", "NM2", "NL2"]

# # Set directory to save RSA files
# RSA_wd = f"{wd}/RSA_files"