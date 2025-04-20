import os
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def train_classifier(combined_conditions_dir, weight_output_dir, time_windows):
    """
    Train ensemble classifier (SVM, LDA, GNB) using Leave-One-Group-Out CV.
    Extract average of SVM and LDA weights for Haufe transformation.
    """
    os.makedirs(weight_output_dir, exist_ok=True)

    for time_window in time_windows:
        logger.info(f"\nTraining classifiers for time window: {time_window}")
        X_all, y_all, groups_all = [], [], []
        subjects = sorted(set(
            fname.split(f"_{time_window}_")[0]
            for fname in os.listdir(combined_conditions_dir)
            if time_window in fname and fname.endswith("_combined.npy")
        ))

        for subject_id in subjects:
            food_path = os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_food_combined.npy")
            nonfood_path = os.path.join(combined_conditions_dir, f"{subject_id}_{time_window}_nonfood_combined.npy")
            if not os.path.exists(food_path) or not os.path.exists(nonfood_path):
                logger.info(f"Skipped {subject_id} — missing files.")
                continue

            food_data = np.load(food_path)
            nonfood_data = np.load(nonfood_path)
            min_trials = min(len(food_data), len(nonfood_data))
            food_data = food_data[np.random.choice(len(food_data), min_trials, replace=False)]
            nonfood_data = nonfood_data[np.random.choice(len(nonfood_data), min_trials, replace=False)]

            X_all.append(np.vstack([food_data, nonfood_data]))
            y_all.append(np.array([1]*min_trials + [0]*min_trials))
            groups_all.append(np.array([subject_id]*(2*min_trials)))

        if len(groups_all) < 2:
            logger.info("Not enough subjects for cross-validation.")
            continue

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        groups = np.concatenate(groups_all)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        base_learners = [
            ("svm", SVC(kernel="linear", probability=True)),
            ("lda", LDA()),
            ("gnb", GaussianNB())
        ]
        ensemble = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
        logo = LeaveOneGroupOut()

        for train_idx, test_idx in logo.split(X, y, groups):
            test_subject = groups[test_idx][0]
            X_train, y_train = X[train_idx], y[train_idx]
            ensemble.fit(X_train, y_train)
            svm_w = ensemble.named_estimators_["svm"].coef_
            lda_w = ensemble.named_estimators_["lda"].coef_
            avg_w = (svm_w + lda_w) / 2
            save_path = os.path.join(weight_output_dir, f"{test_subject}_{time_window}_weights.npy")
            np.save(save_path, avg_w)
            logger.info(f"Saved weights for {test_subject} ({time_window})")
