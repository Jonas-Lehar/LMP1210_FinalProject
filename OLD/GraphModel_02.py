#!/usr/bin/env python3
"""
Semi-supervised LabelSpreading on IDR affinity matrices.
Loops through all matrix/label pairs, aggregates results, and produces
evaluation plots including leave-one-out accuracy, confusion matrix,
per-class confidence distributions, and cluster-sorted heatmaps.

Usage: python GraphModel_02.py
"""

from pathlib import Path
import sys
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR        = Path("./data")
LABELS_DIR      = Path("./Cleaned_Data")
OUTPUT_DIR      = Path("./results")

# LabelSpreading hyperparameters
ALPHA           = 0.5   # clamping factor (0 = hard labels, 1 = fully diffused)
N_NEIGHBOURS    = 7     # KNN kernel neighbours
MAX_ITER        = 100
TOL             = 1e-4
N_JOBS          = -1

# Unknown class handling
CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_LABEL        = 19

# Leave-one-out evaluation: fraction of known labels to mask per run
LOO_FRACTION = 0.2   # hold out 20% of known labels for accuracy estimation
LOO_SEED     = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_affinity_matrix(path: Path):
    df = pd.read_csv(path, index_col=0)
    idr_names = df.index.to_numpy()
    A = df.to_numpy(dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return idr_names, A


def load_known_labels(path: Path, idr_names: np.ndarray, unlabeled_value: int = -1):
    known_df = pd.read_csv(path)
    if "idr" not in known_df.columns or "condensate" not in known_df.columns:
        raise ValueError(f"{path} must have columns 'idr' and 'condensate'.")
    known_map = dict(zip(known_df["idr"], known_df["condensate"]))
    y = np.full(len(idr_names), unlabeled_value, dtype=int)
    for i, name in enumerate(idr_names):
        if name in known_map:
            y[i] = int(known_map[name])
    return y


def run_label_spreading(A, y, alpha=ALPHA, n_neighbors=N_NEIGHBOURS,
                        max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS):
    model = LabelSpreading(
        kernel="knn",
        n_neighbors=n_neighbors,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )
    model.fit(A, y)
    proba   = model.label_distributions_
    classes = model.classes_
    argmax_idx     = proba.argmax(axis=1)
    max_proba      = proba[np.arange(len(proba)), argmax_idx]
    predicted_labels = classes[argmax_idx]
    return model, predicted_labels, max_proba


def derive_final_labels(predicted_labels, max_proba,
                        unknown_label=UNKNOWN_LABEL,
                        confidence_threshold=CONFIDENCE_THRESHOLD):
    final_labels = predicted_labels.copy()
    final_labels[max_proba < confidence_threshold] = unknown_label
    return final_labels


def build_results_df(idr_names, y_known, predicted_labels, final_labels,
                     max_proba, label_distributions, classes):
    result_dict = {
        "idr":         idr_names,
        "known_label": y_known,
        "pred_label":  predicted_labels,
        "final_label": final_labels,
        "max_proba":   max_proba,
    }
    for i, cls in enumerate(classes):
        result_dict[f"prob_{cls}"] = label_distributions[:, i]
    return pd.DataFrame(result_dict)


# ---------------------------------------------------------------------------
# Evaluation: leave-one-out accuracy
# ---------------------------------------------------------------------------

def loo_accuracy(A, y, fraction=LOO_FRACTION, seed=LOO_SEED, **kwargs):
    """
    Mask `fraction` of known labels, run LabelSpreading, report accuracy
    on the held-out set. Returns (accuracy, y_true, y_pred).
    """
    rng = np.random.default_rng(seed)
    labeled_idx = np.where(y != -1)[0]

    if len(labeled_idx) < 5:
        return None, None, None   # too few labels to evaluate

    n_hold = max(1, int(len(labeled_idx) * fraction))
    hold_idx = rng.choice(labeled_idx, size=n_hold, replace=False)

    y_masked = y.copy()
    y_masked[hold_idx] = -1

    _, pred, _ = run_label_spreading(A, y_masked, **kwargs)

    y_true = y[hold_idx]
    y_pred = pred[hold_idx]
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_heatmap(A, results_df, output_path, label_column="final_label", title=""):
    labels = results_df[label_column].to_numpy()
    conf   = results_df["max_proba"].to_numpy()
    order  = np.lexsort((-conf, labels))
    A_sorted = A[order][:, order]
    labels_sorted = labels[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(A_sorted, cmap="viridis", xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Affinity"}, ax=ax)
    ax.set_title(title or f"Affinity matrix sorted by {label_column}")

    _, boundaries = np.unique(labels_sorted, return_index=True)
    for b in boundaries:
        ax.axhline(b, color="white", linewidth=0.5)
        ax.axvline(b, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path, title=""):
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalise by true class counts (row-wise)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"label": "Recall"}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (LOO held-out labels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_by_class(results_df, output_path, title=""):
    """Box plot of max_proba per predicted condensate class."""
    df = results_df[results_df["final_label"] != UNKNOWN_LABEL].copy()
    if df.empty:
        return
    order = sorted(df["final_label"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(order)), 5))
    sns.boxplot(data=df, x="final_label", y="max_proba", order=order, ax=ax)
    ax.axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({CONFIDENCE_THRESHOLD})")
    ax.set_xlabel("Condensate class")
    ax.set_ylabel("Max probability")
    ax.set_title(title or "Prediction confidence by class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_class_distribution(results_df, output_path, batch_name=""):
    """Stacked bar: known vs newly-assigned vs unknown per class."""
    df = results_df.copy()
    df["status"] = "newly assigned"
    df.loc[df["known_label"] != -1, "status"] = "known"
    df.loc[df["final_label"] == UNKNOWN_LABEL, "status"] = "unknown"

    counts = (df.groupby(["final_label", "status"])
                .size()
                .unstack(fill_value=0))
    # Ensure all status columns exist
    for col in ["known", "newly assigned", "unknown"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[["known", "newly assigned", "unknown"]]
    counts.plot(kind="bar", stacked=True, figsize=(max(8, len(counts)), 5),
                color=["steelblue", "orange", "lightgrey"])
    plt.xlabel("Condensate class")
    plt.ylabel("Number of IDRs")
    plt.title(f"Class assignment distribution — {batch_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


HC_MAX_SIZE = 2000  # skip HC for matrices larger than this (too slow)

def plot_hierarchical_vs_label_spreading(A, results_df, output_path, batch_name=""):
    """
    Compare LabelSpreading assignments to hierarchical clustering via ARI.
    Plots a scatter of HC cluster vs LS final_label.
    Skipped for matrices larger than HC_MAX_SIZE.
    """
    n = len(A)
    if n > HC_MAX_SIZE:
        print(f"  Skipping HC comparison for {batch_name} (n={n} > {HC_MAX_SIZE})")
        return None

    # Convert affinity to distance (clip to avoid negatives from float noise)
    D = 1.0 - np.clip(A, 0, 1)
    np.fill_diagonal(D, 0.0)

    n_clusters = len(results_df["final_label"].unique())
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    hc_labels = hc.fit_predict(D)

    ls_labels = results_df["final_label"].to_numpy()
    ari = adjusted_rand_score(hc_labels, ls_labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(hc_labels, ls_labels,
                         c=results_df["max_proba"].to_numpy(),
                         cmap="viridis", alpha=0.4, s=10)
    plt.colorbar(scatter, ax=ax, label="LS confidence")
    ax.set_xlabel("Hierarchical clustering label")
    ax.set_ylabel("LabelSpreading final label")
    ax.set_title(f"{batch_name} — HC vs LS  (ARI = {ari:.3f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return ari


# ---------------------------------------------------------------------------
# Pair data files with label files
# ---------------------------------------------------------------------------

def find_pairs():
    """
    Match data/X_Y.csv  <->  Cleaned_Data/X_Y_test.csv
    Returns list of (affinity_path, labels_path, batch_name).
    """
    pairs = []
    for aff_path in sorted(DATA_DIR.glob("*.csv")):
        if "test" in aff_path.name:
            continue
        stem = aff_path.stem          # e.g. "201_250"
        label_path = LABELS_DIR / f"{stem}_test.csv"
        if label_path.is_file():
            pairs.append((aff_path, label_path, stem))
        else:
            print(f"WARNING: no label file found for {aff_path.name}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    pairs = find_pairs()
    if not pairs:
        raise RuntimeError("No matching data/label pairs found.")

    all_results   = []
    loo_y_true_all = []
    loo_y_pred_all = []
    ari_scores     = {}

    for aff_path, label_path, batch in pairs:
        print(f"\n{'='*60}")
        print(f"Batch: {batch}")
        print(f"  Affinity : {aff_path}")
        print(f"  Labels   : {label_path}")

        idr_names, A = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)

        n_labeled = (y != -1).sum()
        print(f"  IDRs: {len(idr_names)}  |  Labeled: {n_labeled}")

        # --- Leave-one-out accuracy ---
        acc, yt, yp = loo_accuracy(A, y, alpha=ALPHA, n_neighbors=N_NEIGHBOURS,
                                   max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
        if acc is not None:
            print(f"  LOO accuracy ({int(LOO_FRACTION*100)}% held out): {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())

            plot_confusion_matrix(
                yt, yp,
                OUTPUT_DIR / f"{batch}_confusion.png",
                title=f"{batch} — LOO confusion matrix (acc={acc:.2f})"
            )
        else:
            print("  Skipping LOO (too few labeled samples)")

        # --- Full run ---
        model, predicted_labels, max_proba = run_label_spreading(A, y)
        final_labels = derive_final_labels(predicted_labels, max_proba)

        results_df = build_results_df(
            idr_names, y, predicted_labels, final_labels,
            max_proba, model.label_distributions_, model.classes_
        )
        results_df["batch"] = batch
        all_results.append(results_df)

        # --- Per-batch plots ---
        plot_heatmap(A, results_df,
                     OUTPUT_DIR / f"{batch}_heatmap.png",
                     title=f"{batch} — affinity matrix sorted by final label")

        plot_confidence_by_class(results_df,
                                 OUTPUT_DIR / f"{batch}_confidence.png",
                                 title=f"{batch} — confidence by class")

        plot_class_distribution(results_df,
                                OUTPUT_DIR / f"{batch}_class_dist.png",
                                batch_name=batch)

        ari = plot_hierarchical_vs_label_spreading(
            A, results_df,
            OUTPUT_DIR / f"{batch}_hc_vs_ls.png",
            batch_name=batch
        )
        ari_scores[batch] = ari
        if ari is not None:
            print(f"  ARI (HC vs LS): {ari:.3f}")

    # --- Aggregate results ---
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_csv = OUTPUT_DIR / "all_results.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nWrote combined results to {combined_csv}")

    # --- Global confusion matrix across all batches ---
    if loo_y_true_all:
        plot_confusion_matrix(
            np.array(loo_y_true_all),
            np.array(loo_y_pred_all),
            OUTPUT_DIR / "global_confusion.png",
            title="Global LOO confusion matrix (all batches)"
        )
        global_acc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"\nGlobal LOO accuracy: {global_acc:.3f}")

    # --- ARI summary ---
    print("\nARI scores (Hierarchical Clustering vs LabelSpreading):")
    for batch, ari in ari_scores.items():
        if ari is not None:
            print(f"  {batch}: {ari:.3f}")
        else:
            print(f"  {batch}: skipped (matrix too large)")

    # --- Global class distribution ---
    plot_class_distribution(
        combined_df,
        OUTPUT_DIR / "global_class_dist.png",
        batch_name="all batches"
    )


if __name__ == "__main__":
    main()
