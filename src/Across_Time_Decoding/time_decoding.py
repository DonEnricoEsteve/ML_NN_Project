
import numpy as np
from sklearn.base import BaseEstimator
from mne.decoding import SlidingEstimator, cross_val_multiscore, GeneralizingEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def stacking_classifier_defenition(time_decoding_classifier: type[SlidingEstimator] | type[GeneralizingEstimator],
base_classifiers: list[tuple[str, BaseEstimator]] = 
[
    ('svm', SVC(probability=True, random_state=42, kernel='rbf')),
    ('lda', LDA()),
    ('gnb', GaussianNB())
], 
final_classifier: BaseEstimator = LogisticRegression(random_state=42)) -> type[SlidingEstimator] | type[GeneralizingEstimator]:

    """
    Function defines the classifier including ensemble stacked classifiers and time_decoding wrapper.

    Parameters:
    * base_classifiers - a list of first level classifiers with the desired parameters.
    * final_classifier - a secondary level classifier with the desired parameters.
    * time_decoding_classifier - the time decoding estimator wrapper.

    Returns: 
    time_decod - the defined classifier ready for fitting and testing.

    """
    # Create a stacking ensemble using the specified base classifiers and a final classifier
    stacking_clf = StackingClassifier(
        estimators=base_classifiers,           # First-level estimators in the ensemble
        final_estimator=final_classifier       # Meta-classifier that combines the base models
    )

    # Wrap the stacking classifier in a temporal decoding estimator (SlidingEstimator or GeneralizingEstimator)
    time_decod = time_decoding_classifier(base_estimator=stacking_clf, n_jobs=None, scoring="roc_auc", verbose=True)

    # Return the fully configured time decoding model
    return time_decod


def cross_val_iter(X: np.ndarray, y: np.ndarray, classifier: BaseEstimator, random_state: int) -> np.ndarray:
    """
    Function preforms cross validation on data for a single itteration.

    Parameters:
    * X - the data to be decoded
    * y - the data labels
    * classifier - defined classifier for fitting and testing
    * random_state - the random_state for data shuffling prior to K-Fold cross validation

    Returns: 
    mean_cv_scores - the mean of the scores recieved in cross validation

    """
    # Define a Stratified K-Fold cross-validator to preserve class proportions in each fold
    cv = StratifiedKFold(shuffle=True, random_state=random_state)

    # Run cross-validation and get scores for each fold 
    scores = cross_val_multiscore(classifier, X=X, y=y, cv=cv, n_jobs=None)

    # Compute the mean across the k-folds for each time point
    mean_cv_scores = np.mean(scores, axis=0)  

    return  mean_cv_scores
