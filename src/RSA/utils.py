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

# ============================================================ #
# Part 5: Representational Similarity Analysis (RSA)
# ============================================================ #

# Set directory to save RSA files
RSA_wd = f"{wd}/RSA_Results"

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
