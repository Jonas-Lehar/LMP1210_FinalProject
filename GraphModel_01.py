#!/usr/bin/env python3
"""
Semi-supervised LabelSpreading on an IDR affinity matrix (RBF kernel)
and cluster-sorted affinity heatmap visualization.

Usage
-----
Edit the configuration section at the top of this file:

    AFFINITY_CSV = "idr_affinity.csv"
    KNOWN_LABELS_CSV = "known_assignments.csv"

Then run the script in your Python environment.
"""



from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelSpreading  # [web:4][web:12]



# Configuration (edit these paths / hyperparameters as needed)
# AFFINITY_CSV = ".\\data\\401_450.csv"          # N x N affinity matrix CSV
# KNOWN_LABELS_CSV = ".\\data\\401_450_test.csv" # CSV with columns 'idr', 'condensate'

# For MacOS Users (or Linux)
AFFINITY_CSV = "./data/401_450.csv"          # N x N affinity matrix CSV
KNOWN_LABELS_CSV = "./data/401_450_test.csv" # CSV with columns 'idr', 'condensate'

# LabelSpreading hyperparameters
ALPHA = 0.5          # clamping factor (0 = hard labels, 1 = fully diffused)
GAMMA = 7         # RBF gamma (higher values = more localized affinity)

N_NEIGHBOURS = 7     # for KNN kernel (if used instead of RBF)

MAX_ITER = 100
TOL = 1e-4
N_JOBS = -1          # use all cores

# Unknown class handling
CONFIDENCE_THRESHOLD = 0.6  # min max_proba to accept a condensate label
UNKNOWN_LABEL = 19          # integer value for "unknown" condensate

# Outputs
RESULTS_CSV = "idr_labelspreading_results.csv"
HEATMAP_PNG = "idr_affinity_heatmap_sorted.png"



# Functions
def load_affinity_matrix(path: Path):
    """Load N x N affinity matrix with IDR names on rows and columns."""
    df = pd.read_csv(path, index_col=0)
    idr_names = df.index.to_numpy()

    # Basic sanity check: columns should match index
    if not np.array_equal(idr_names, df.columns.to_numpy()):
        print(
            "WARNING: Row and column labels for the affinity matrix do not match exactly.\n"
            "         Proceeding anyway, but you should verify the input.",
            file=sys.stderr,
        )

    A = df.to_numpy(dtype=float)

    # Enforce symmetry and zero diagonal
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    return idr_names, A


def load_known_labels(path: Path, idr_names: np.ndarray, unlabeled_value: int = -1):
    """
    Build label vector y of length N from known_assignments.csv.

    Parameters
    ----------
    path : Path
        Path to known assignments CSV with columns 'idr' and 'condensate'.
    idr_names : np.ndarray
        Array of IDR names, same order/length as affinity matrix.
    unlabeled_value : int
        Value used to mark unlabeled samples (-1 as required by sklearn).[web:24]

    Returns
    -------
    y : np.ndarray of shape (N,)
        Labels for each IDR (>=0 for labeled, unlabeled_value for unlabeled).
    """
    known_df = pd.read_csv(path)

    if "idr" not in known_df.columns or "condensate" not in known_df.columns:
        raise ValueError(
            f"Known labels file {path} must have columns 'idr' and 'condensate'."
        )

    # Map from IDR name to condensate integer label
    known_map = dict(zip(known_df["idr"], known_df["condensate"]))

    y = np.full(len(idr_names), unlabeled_value, dtype=int)

    for i, name in enumerate(idr_names):
        if name in known_map:
            # Assumes condensate labels already in {1, ..., 18}
            y[i] = int(known_map[name])

    # Report any entries in known_df not found in idr_names
    known_idrs = set(known_df["idr"])
    present_idrs = set(idr_names)
    missing_in_matrix = sorted(list(known_idrs - present_idrs))
    if missing_in_matrix:
        print(
            f"WARNING: {len(missing_in_matrix)} IDRs in known_assignments.csv "
            f"are not present in the affinity matrix header. "
            f"Example: {missing_in_matrix[:5]}",
            file=sys.stderr,
        )
    return y


def run_label_spreading_rbf(
    A: np.ndarray,
    y: np.ndarray,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    N_NEIGHBOURS: int = N_NEIGHBOURS,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_jobs: int | None = -1,
):
    """
    Run sklearn.semi_supervised.LabelSpreading with RBF/KNN kernel.

    Uses rows of A as feature vectors (X = A).

    Returns
    -------
    model : fitted LabelSpreading instance
    predicted_labels : np.ndarray of shape (N,)
        Argmax class for each IDR (over non-negative classes).
    max_proba : np.ndarray of shape (N,)
        Maximum class probability per IDR.
    """
    X = A  # (n_samples, n_features)

    model = LabelSpreading(
        #kernel="rbf",
        #gamma=gamma,
        kernel="knn",
        n_neighbors=N_NEIGHBOURS,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )
    model.fit(X, y)  # y must contain -1 for unlabeled samples.[web:24]

    proba = model.label_distributions_  # (N, C)
    classes = model.classes_            # length C

    argmax_idx = proba.argmax(axis=1)
    max_proba = proba[np.arange(len(proba)), argmax_idx]
    predicted_labels = classes[argmax_idx]

    return model, predicted_labels, max_proba


def derive_final_labels(
    predicted_labels: np.ndarray,
    max_proba: np.ndarray,
    unknown_label: int = 19,
    confidence_threshold: float = 0.6,
):
    """
    Convert predicted_labels + max_proba into final labels with an explicit
    'unknown' class when confidence < threshold.

    Returns
    -------
    final_labels : np.ndarray of shape (N,)
    """
    final_labels = predicted_labels.copy()
    low_conf_mask = max_proba < confidence_threshold
    final_labels[low_conf_mask] = unknown_label
    return final_labels


def build_results_dataframe(
    idr_names: np.ndarray,
    y_known: np.ndarray,
    predicted_labels: np.ndarray,
    final_labels: np.ndarray,
    max_proba: np.ndarray,
) -> pd.DataFrame:
    """Assemble all results into a DataFrame."""
    return pd.DataFrame(
        {
            "idr": idr_names,
            "known_label": y_known,
            "pred_label": predicted_labels,
            "final_label": final_labels,
            "max_proba": max_proba,
        }
    )


def plot_cluster_sorted_heatmap(
    A: np.ndarray,
    results_df: pd.DataFrame,
    output_path: Path,
    label_column: str = "final_label",
):
    """
    Plot affinity matrix sorted by the chosen label column, then by confidence
    (max_proba), and save to output_path.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Symmetric affinity matrix.
    results_df : pd.DataFrame
        Must contain columns label_column and "max_proba".
    output_path : Path
        PNG path to write.
    label_column : str
        Column in results_df to sort by (e.g. 'final_label' or 'pred_label').
    """
    labels = results_df[label_column].to_numpy()
    conf = results_df["max_proba"].to_numpy()

    # Sort by label, then by descending confidence within label
    order = np.lexsort((-conf, labels))

    A_sorted = A[order][:, order]
    labels_sorted = labels[order]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        A_sorted,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Affinity"},
    )
    ax.set_title(f"Affinity matrix sorted by {label_column}")

    # Overlay boundaries between different labels
    unique_labels, boundaries = np.unique(labels_sorted, return_index=True)
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=0.5)
        ax.axvline(b, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def build_results_dataframe(
    idr_names: np.ndarray,
    y_known: np.ndarray,
    predicted_labels: np.ndarray,
    final_labels: np.ndarray,
    max_proba: np.ndarray,
    label_distributions: np.ndarray,
    classes: np.ndarray,
    unknown_label: int = 19,
) -> pd.DataFrame:
    """
    Assemble all results into a DataFrame with all class probabilities.
    
    Parameters
    ----------
    idr_names : np.ndarray
        IDR identifiers.
    y_known : np.ndarray
        Original known labels (with -1 for unlabeled).
    predicted_labels : np.ndarray
        Argmax predicted label for each IDR.
    final_labels : np.ndarray
        Final labels with unknown class applied.
    max_proba : np.ndarray
        Maximum probability across classes for each IDR.
    label_distributions : np.ndarray, shape (N, n_classes)
        Probability distributions from LabelSpreading.
    classes : np.ndarray
        Array of class labels from model.classes_.
    unknown_label : int
        The integer used for unknown condensate.
        
    Returns
    -------
    pd.DataFrame with columns:
        - idr
        - known_label
        - pred_label
        - final_label
        - max_proba
        - prob_1, prob_2, ..., prob_18, prob_19
    """
    # Start with basic info columns
    result_dict = {
        "idr": idr_names,
        "known_label": y_known,
        "pred_label": predicted_labels,
        "final_label": final_labels,
        "max_proba": max_proba,
    }
    
    # Add probability columns for condensates 1-18
    # label_distributions has shape (N, len(classes))
    # classes contains the labels that were present in training (e.g., [1, 2, ..., 18])
    for i, cls in enumerate(classes):
        result_dict[f"prob_{cls}"] = label_distributions[:, i]
    
    # Add prob_19 (unknown) column
    # IDRs assigned to unknown have low confidence in all known classes
    # We'll set prob_19 = 1 - max_proba for those assigned to unknown
    # and 0 for those confidently assigned to a known condensate
    prob_unknown = np.zeros(len(idr_names))
    unknown_mask = final_labels == unknown_label
    prob_unknown[unknown_mask] = 1.0 - max_proba[unknown_mask]
    
    result_dict[f"prob_{unknown_label}"] = prob_unknown
    
    return pd.DataFrame(result_dict)
    print(f"Wrote heatmap to {heatmap_path}")



# Main execution block
def main():
    # Resolve paths
    affinity_path = Path(AFFINITY_CSV)
    labels_path = Path(KNOWN_LABELS_CSV)

    if not affinity_path.is_file():
        raise FileNotFoundError(f"Affinity CSV not found: {affinity_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Known labels CSV not found: {labels_path}")

    # Load data
    idr_names, A = load_affinity_matrix(affinity_path)
    y = load_known_labels(labels_path, idr_names, unlabeled_value=-1)

    # Run LabelSpreading (RBF kernel)
    model, predicted_labels, max_proba = run_label_spreading_rbf(
        A,
        y,
        alpha=ALPHA,
        gamma=GAMMA,
        max_iter=MAX_ITER,
        tol=TOL,
        n_jobs=N_JOBS,
    )

    # Derive final labels with explicit unknown category
    final_labels = derive_final_labels(
        predicted_labels,
        max_proba,
        unknown_label=UNKNOWN_LABEL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # Build results DataFrame with all class probabilities
    results_df = build_results_dataframe(
        idr_names, 
        y, 
        predicted_labels, 
        final_labels, 
        max_proba,
        model.label_distributions_,  # Add this
        model.classes_,              # Add this
        unknown_label=UNKNOWN_LABEL,
    )
    
    results_csv_path = Path(RESULTS_CSV)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Wrote results to {results_csv_path}")

    # Plot cluster-sorted heatmap
    heatmap_path = Path(HEATMAP_PNG)
    plot_cluster_sorted_heatmap(
        A,
        results_df,
        output_path=heatmap_path,
        label_column="final_label",
    )



if __name__ == "__main__":
    main()
