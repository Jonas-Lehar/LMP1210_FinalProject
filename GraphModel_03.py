#!/usr/bin/env python3
"""
Semi-supervised LabelSpreading on IDR affinity matrices.
Loops through all matrix/label pairs, aggregates results, and produces
evaluation plots including leave-one-out accuracy, confusion matrix,
per-class confidence distributions, and cluster-sorted heatmaps.

Usage: python GraphModel_03.py
"""

from pathlib import Path
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, adjusted_rand_score, f1_score
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
LOO_FRACTION = 0.2   # hold out 20% of known labels per repeat
LOO_SEED     = 42
LOO_REPEATS  = 5     # number of random splits to average over (reduces noise)

# ---------------------------------------------------------------------------
# HC size limit
# Set SKIP_HC_LARGE = True  to skip HC for matrices with n > HC_MAX_SIZE
#                            (faster, good for running through Kiro/CI)
# Set SKIP_HC_LARGE = False to always run HC regardless of matrix size
#                            (slower, use when running locally for full results)
# ---------------------------------------------------------------------------
SKIP_HC_LARGE = False
HC_MAX_SIZE   = 2000

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# Set RUN_HYPERPARAMETER_TUNING = True to run the grid search before the
# main loop. Results are saved to OUTPUT_DIR/hyperparam_tuning.csv and a
# heatmap is saved to OUTPUT_DIR/hyperparam_heatmap.png.
#
# Tuning uses LOO accuracy averaged across all batches that have enough
# labeled samples (>= MIN_LABELS_FOR_TUNING).
# ---------------------------------------------------------------------------
RUN_HYPERPARAMETER_TUNING = True

TUNE_ALPHAS      = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
TUNE_N_NEIGHBORS = [3, 5, 7, 10, 15, 20]
# Optionally also tune the kernel; set to ["knn"] to keep it fixed
TUNE_KERNELS     = ["knn"]   # add "rbf" here if you want to compare kernels
TUNE_GAMMAS      = [1, 3, 5, 7, 10]   # only used when "rbf" is in TUNE_KERNELS

MIN_LABELS_FOR_TUNING = 10   # skip batches with fewer known labels when tuning


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
                        kernel="knn", gamma=5,
                        max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS):
    kwargs = dict(
        kernel=kernel,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
    )
    if kernel == "knn":
        kwargs["n_neighbors"] = n_neighbors
    elif kernel == "rbf":
        kwargs["gamma"] = gamma

    model = LabelSpreading(**kwargs)
    model.fit(A, y)
    proba   = model.label_distributions_
    classes = model.classes_
    argmax_idx       = proba.argmax(axis=1)
    max_proba        = proba[np.arange(len(proba)), argmax_idx]
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

def loo_accuracy(A, y, fraction=LOO_FRACTION, seed=LOO_SEED,
                 n_repeats=LOO_REPEATS, **kwargs):
    """
    Repeated random holdout evaluation.

    Masks `fraction` of known labels `n_repeats` times (different random
    splits each time), runs LabelSpreading on each, and returns the mean
    macro-F1 and accuracy across repeats.

    Using macro-F1 as the primary metric because class 6 dominates the label
    distribution (~76% of labeled IDRs). Accuracy alone would reward a model
    that just predicts class 6 for everything. Macro-F1 weights all classes
    equally regardless of size.

    Returns (mean_f1, mean_acc, y_true_all, y_pred_all)
        where y_true_all / y_pred_all are concatenated across all repeats
        (useful for building a confusion matrix).
    """
    labeled_idx = np.where(y != -1)[0]

    if len(labeled_idx) < 5:
        return None, None, None, None

    f1s, accs = [], []
    y_true_all, y_pred_all = [], []

    for rep in range(n_repeats):
        rng      = np.random.default_rng(seed + rep)
        n_hold   = max(1, int(len(labeled_idx) * fraction))
        hold_idx = rng.choice(labeled_idx, size=n_hold, replace=False)

        y_masked = y.copy()
        y_masked[hold_idx] = -1

        _, pred, _ = run_label_spreading(A, y_masked, **kwargs)

        y_true = y[hold_idx]
        y_pred = pred[hold_idx]

        # macro-F1: unweighted mean across classes present in y_true
        f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = (y_true == y_pred).mean()

        f1s.append(f1)
        accs.append(acc)
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    return float(np.mean(f1s)), float(np.mean(accs)), \
           np.array(y_true_all), np.array(y_pred_all)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(pairs):
    """
    Grid search over TUNE_ALPHAS x TUNE_N_NEIGHBORS (x TUNE_KERNELS).
    For each combination, runs LOO accuracy on every batch that has at least
    MIN_LABELS_FOR_TUNING known labels, then averages across batches.

    Saves results to OUTPUT_DIR/hyperparam_tuning.csv and plots a heatmap
    (one per kernel) to OUTPUT_DIR/hyperparam_heatmap_<kernel>.png.

    Returns the best (alpha, n_neighbors, kernel, gamma) tuple.
    """
    print("\n" + "="*60)
    print("Hyperparameter tuning")
    print("="*60)

    # Pre-load all matrices and labels (skip batches with too few labels)
    datasets = []
    for aff_path, label_path, batch in pairs:
        idr_names, A = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        n_labeled = (y != -1).sum()
        if n_labeled < MIN_LABELS_FOR_TUNING:
            print(f"  Skipping {batch} for tuning (only {n_labeled} labeled samples)")
            continue
        datasets.append((batch, A, y))

    if not datasets:
        print("  No batches with enough labels for tuning. Skipping.")
        return ALPHA, N_NEIGHBOURS, "knn", 5

    records = []

    # Build full parameter grid
    param_grid = []
    for kernel in TUNE_KERNELS:
        if kernel == "knn":
            for alpha, k in itertools.product(TUNE_ALPHAS, TUNE_N_NEIGHBORS):
                param_grid.append({"kernel": kernel, "alpha": alpha,
                                   "n_neighbors": k, "gamma": None})
        elif kernel == "rbf":
            for alpha, gamma in itertools.product(TUNE_ALPHAS, TUNE_GAMMAS):
                param_grid.append({"kernel": kernel, "alpha": alpha,
                                   "n_neighbors": None, "gamma": gamma})

    total = len(param_grid)
    for idx, params in enumerate(param_grid):
        kernel = params["kernel"]
        alpha  = params["alpha"]
        k      = params["n_neighbors"]
        gamma  = params["gamma"]

        label = (f"kernel={kernel}, alpha={alpha}, "
                 + (f"k={k}" if kernel == "knn" else f"gamma={gamma}"))

        accs = []
        f1s  = []
        weights = []
        for batch, A, y in datasets:
            f1, acc, _, _ = loo_accuracy(
                A, y,
                alpha=alpha,
                n_neighbors=k if k is not None else N_NEIGHBOURS,
                kernel=kernel,
                gamma=gamma if gamma is not None else 5,
                max_iter=MAX_ITER,
                tol=TOL,
                n_jobs=N_JOBS,
            )
            if f1 is not None:
                n_labeled = int((y != -1).sum())
                f1s.append(f1)
                accs.append(acc)
                weights.append(n_labeled)

        # Weighted mean by number of labeled samples per batch
        if f1s:
            w = np.array(weights, dtype=float)
            mean_f1  = float(np.average(f1s,  weights=w))
            mean_acc = float(np.average(accs, weights=w))
        else:
            mean_f1 = mean_acc = float("nan")

        print(f"  [{idx+1}/{total}] {label}  ->  macro-F1 = {mean_f1:.3f}  acc = {mean_acc:.3f}")
        records.append({**params, "mean_macro_f1": mean_f1, "mean_acc": mean_acc})

    results_df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "hyperparam_tuning.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved tuning results to {csv_path}")

    # Plot heatmap(s) — one per kernel
    for kernel in TUNE_KERNELS:
        sub = results_df[results_df["kernel"] == kernel]
        if kernel == "knn":
            pivot = sub.pivot(index="alpha", columns="n_neighbors",
                              values="mean_macro_f1")
            xlabel, ylabel = "n_neighbors", "alpha"
        else:
            pivot = sub.pivot(index="alpha", columns="gamma",
                              values="mean_macro_f1")
            xlabel, ylabel = "gamma", "alpha"

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)), max(4, len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                    cbar_kws={"label": "Weighted mean macro-F1"}, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Hyperparameter tuning — kernel={kernel}\n"
                     f"(macro-F1, {LOO_REPEATS} repeats, weighted by labeled count)")
        plt.tight_layout()
        heatmap_path = OUTPUT_DIR / f"hyperparam_heatmap_{kernel}.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"  Saved tuning heatmap to {heatmap_path}")

    # Pick best params by macro-F1
    best_row = results_df.loc[results_df["mean_macro_f1"].idxmax()]
    best_alpha  = best_row["alpha"]
    best_k      = best_row["n_neighbors"]
    best_kernel = best_row["kernel"]
    best_gamma  = best_row["gamma"]
    print(f"\n  Best params: kernel={best_kernel}, alpha={best_alpha}, "
          + (f"k={best_k}" if best_kernel == "knn" else f"gamma={best_gamma}")
          + f"  (macro-F1 = {best_row['mean_macro_f1']:.3f}, acc = {best_row['mean_acc']:.3f})")

    return best_alpha, best_k, best_kernel, best_gamma


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_heatmap(A, results_df, output_path, label_column="final_label", title=""):
    labels = results_df[label_column].to_numpy()
    conf   = results_df["max_proba"].to_numpy()
    order  = np.lexsort((-conf, labels))
    A_sorted      = A[order][:, order]
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
    df = results_df.copy()
    df["status"] = "newly assigned"
    df.loc[df["known_label"] != -1, "status"] = "known"
    df.loc[df["final_label"] == UNKNOWN_LABEL, "status"] = "unknown"

    counts = (df.groupby(["final_label", "status"])
                .size()
                .unstack(fill_value=0))
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


def plot_hierarchical_vs_label_spreading(A, results_df, output_path, batch_name=""):
    """
    Compare LabelSpreading assignments to hierarchical clustering via ARI.
    Skipped for large matrices when SKIP_HC_LARGE is True.
    """
    n = len(A)
    if SKIP_HC_LARGE and n > HC_MAX_SIZE:
        print(f"  Skipping HC comparison for {batch_name} (n={n} > {HC_MAX_SIZE}, SKIP_HC_LARGE=True)")
        return None

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
    pairs = []
    for aff_path in sorted(DATA_DIR.glob("*.csv")):
        if "test" in aff_path.name:
            continue
        stem       = aff_path.stem
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

    # --- Optional hyperparameter tuning ---
    # Runs before the main loop and updates ALPHA / N_NEIGHBOURS if enabled.
    alpha      = ALPHA
    n_neighbors = N_NEIGHBOURS
    kernel     = "knn"
    gamma      = 5

    if RUN_HYPERPARAMETER_TUNING:
        alpha, n_neighbors, kernel, gamma = tune_hyperparameters(pairs)
        print(f"\nUsing tuned params: kernel={kernel}, alpha={alpha}, "
              + (f"n_neighbors={n_neighbors}" if kernel == "knn" else f"gamma={gamma}"))
    else:
        print(f"\nUsing fixed params: kernel={kernel}, alpha={alpha}, n_neighbors={n_neighbors}")

    all_results    = []
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
        f1, acc, yt, yp = loo_accuracy(
            A, y,
            alpha=alpha, n_neighbors=n_neighbors, kernel=kernel, gamma=gamma,
            max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS,
        )
        if f1 is not None:
            print(f"  LOO macro-F1 ({LOO_REPEATS} repeats, {int(LOO_FRACTION*100)}% held out): {f1:.3f}  acc: {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())
            plot_confusion_matrix(
                yt, yp,
                OUTPUT_DIR / f"{batch}_confusion.png",
                title=f"{batch} — LOO confusion matrix (F1={f1:.2f}, acc={acc:.2f})"
            )
        else:
            print("  Skipping LOO (too few labeled samples)")

        # --- Full run ---
        model, predicted_labels, max_proba = run_label_spreading(
            A, y, alpha=alpha, n_neighbors=n_neighbors,
            kernel=kernel, gamma=gamma, max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS,
        )
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
            batch_name=batch,
        )
        ari_scores[batch] = ari
        if ari is not None:
            print(f"  ARI (HC vs LS): {ari:.3f}")

    # --- Aggregate results ---
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_csv = OUTPUT_DIR / "all_results.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nWrote combined results to {combined_csv}")

    # --- Global confusion matrix ---
    if loo_y_true_all:
        plot_confusion_matrix(
            np.array(loo_y_true_all),
            np.array(loo_y_pred_all),
            OUTPUT_DIR / "global_confusion.png",
            title="Global LOO confusion matrix (all batches)"
        )
        global_f1  = f1_score(loo_y_true_all, loo_y_pred_all, average="macro", zero_division=0)
        global_acc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"\nGlobal LOO macro-F1: {global_f1:.3f}  acc: {global_acc:.3f}")

    # --- ARI summary ---
    print("\nARI scores (Hierarchical Clustering vs LabelSpreading):")
    for batch, ari in ari_scores.items():
        if ari is not None:
            print(f"  {batch}: {ari:.3f}")
        else:
            print(f"  {batch}: skipped (SKIP_HC_LARGE=True or matrix too large)")

    # --- Global class distribution ---
    plot_class_distribution(
        combined_df,
        OUTPUT_DIR / "global_class_dist.png",
        batch_name="all batches"
    )


if __name__ == "__main__":
    main()
