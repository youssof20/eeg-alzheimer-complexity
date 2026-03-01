"""
Random Forest classification on subject-level complexity features.
Metrics per class, confusion matrix, feature importance, predict for new subject.
"""
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

FEATURE_NAMES = ["SampEn", "HFD", "DFA"]
GROUP_ORDER = ["Healthy", "FTD", "Alzheimer's"]


def train_random_forest(df: pd.DataFrame, target_col: str = "group", feature_cols: list = None):
    """
    Train RF on df. Columns must include target_col and feature_cols.
    Returns fitted model, X, y, class names.
    """
    if not SKLEARN_AVAILABLE:
        return None, None, None, []
    feature_cols = feature_cols or FEATURE_NAMES
    X = df[feature_cols].values
    y = df[target_col].values
    classes = sorted(df[target_col].unique().tolist(), key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X, y)
    return clf, X, y, classes


def confusion_matrix_and_metrics(clf, X, y, classes: list):
    """Compute confusion matrix and per-class sensitivity, specificity, plus accuracy."""
    if not SKLEARN_AVAILABLE or clf is None:
        return None, {}, 0.0
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=classes)
    acc = accuracy_score(y, y_pred)
    metrics = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[c] = {"sensitivity": sens, "specificity": spec}
    return cm, metrics, acc


def feature_importance(clf) -> dict:
    """Return dict feature_name -> importance from fitted RF."""
    if not SKLEARN_AVAILABLE or clf is None:
        return {}
    names = FEATURE_NAMES
    if hasattr(clf, "feature_names_in_"):
        names = list(clf.feature_names_in_)
    return dict(zip(names, clf.feature_importances_.tolist()))


def predict_subject(clf, features_dict: dict, classes: list):
    """Predict class and probabilities for one subject. features_dict has SampEn, HFD, DFA."""
    if not SKLEARN_AVAILABLE or clf is None:
        return None, {}
    x = np.array([[features_dict.get("SampEn", np.nan), features_dict.get("HFD", np.nan), features_dict.get("DFA", np.nan)]])
    if np.any(np.isnan(x)):
        return None, {}
    pred = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]
    return pred, dict(zip(classes, proba.tolist()))
