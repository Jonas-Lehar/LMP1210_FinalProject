#!/usr/bin/env python3
"""
GraphModel_05_Normalization.py

LabelSpreading with affinity matrix row-normalization variants.
Tries StandardScaler and RobustScaler (applied row-wise to the affinity
matrix) before running LabelSpreading. A small hyperparameter tuning loop
covers: scaler x alpha x n_neighbors.

Usage: python GraphModel_05_Normalization.py
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path("./data")
LABELS_DIR = Path("./Cleaned_Data")
OUTPUT_DIR = Path("./results_norm")

# Fixed hyperparameters (overridden by tuning if enabled)
SCALER_NAME  = "minmax"   # "standard" | "robust" | "minmax"
ALPHA        = 0.1
N_NEIGHBOURS = 20
MAX_ITER     = 1000
TOL          = 1e-4
N_JOBS       = -1

CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_LABEL        = 19

LOO_FRACTION = 0.15
LOO_SEED     = 42
LOO_REPEATS  = 10

# ---------------------------------------------------------------------------
# Tuning config — keep small so it finishes in reasonable time
# ---------------------------------------------------------------------------
RUN_HYPERPARAMETER_TUNING = True

TUNE_SCALERS     = ["standard", "robust", "minmax"]
TUNE_ALPHAS      = [0.1, 0.3, 0.7, 0.9]
TUNE_N_NEIGHBORS = [5, 10, 20]
MIN_LABELS_FOR_TUNING = 10


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_affinity(A: np.ndarray, scaler_name: str) -> np.ndarray:
    """
    Apply row-wise normalization to the affinity matrix, then re-symmetrize
    and clip to [0, 1].

    StandardScaler: zero-mean, unit-variance per row
    RobustScaler:   median-centered, IQR-scaled per row (robust to outliers)
    MinMaxScaler:   rescale each row to [0, 1]  (preserves relative order)
    """
    scalers = {
        "standard": StandardScaler(),
        "robust":   RobustScaler(),
        "minmax":   MinMaxScaler(),
    }
    if scaler_name not in scalers:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Choose from {list(scalers)}")

    scaler = scalers[scaler_name]
    # Fit on rows (each IDR's affinity profile is one sample)
    A_norm = scaler.fit_transform(A)
    # Re-symmetrize and clip
    A_norm = 0.5 * (A_norm + A_norm.T)
    A_norm = np.clip(A_norm, 0.0, 1.0)
    np.fill_diagonal(A_norm, 0.0)
    return A_norm


# ---------------------------------------------------------------------------
# Helpers (identical to GraphModel_03)
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
        kernel="knn", n_neighbors=int(n_neighbors),
        alpha=alpha, max_iter=max_iter, tol=tol, n_jobs=n_jobs,
    )
    model.fit(A, y)
    proba            = model.label_distributions_
    classes          = model.classes_
    argmax_idx       = proba.argmax(axis=1)
    max_proba        = proba[np.arange(len(proba)), argmax_idx]
    predicted_labels = classes[argmax_idx]
    return model, predicted_labels, max_proba


def derive_final_labels(predicted_labels, max_proba):
    final_labels = predicted_labels.copy()
    final_labels[max_proba < CONFIDENCE_THRESHOLD] = UNKNOWN_LABEL
    return final_labels


def build_results_df(idr_names, y_known, predicted_labels, final_labels,
                     max_proba, label_distributions, classes):
    d = {"idr": idr_names, "known_label": y_known,
         "pred_label": predicted_labels, "final_label": final_labels,
         "max_proba": max_proba}
    for i, cls in enumerate(classes):
        d[f"prob_{cls}"] = label_distributions[:, i]
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
# LOO evaluation
# ---------------------------------------------------------------------------

def loo_accuracy(A, y, fraction=LOO_FRACTION, seed=LOO_SEED,
                 n_repeats=LOO_REPEATS, **kwargs):
    labeled_idx = np.where(y != -1)[0]
    if len(labeled_idx) < 5:
        return None, None, None, None

    f1s, accs, y_true_all, y_pred_all = [], [], [], []
    for rep in range(n_repeats):
        rng      = np.random.default_rng(seed + rep)
        n_hold   = max(1, int(len(labeled_idx) * fraction))
        hold_idx = rng.choice(labeled_idx, size=n_hold, replace=False)
        y_masked = y.copy()
        y_masked[hold_idx] = -1
        _, pred, _ = run_label_spreading(A, y_masked, **kwargs)
        y_true = y[hold_idx]
        y_pred = pred[hold_idx]
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        accs.append((y_true == y_pred).mean())
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())

    return float(np.mean(f1s)), float(np.mean(accs)), \
           np.array(y_true_all), np.array(y_pred_all)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(pairs):
    print("\n" + "="*60)
    print("Hyperparameter tuning — normalization variants")
    print("="*60)

    datasets = []
    for aff_path, label_path, batch in pairs:
        idr_names, A = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        if (y != -1).sum() < MIN_LABELS_FOR_TUNING:
            print(f"  Skipping {batch} for tuning (too few labels)")
            continue
        datasets.append((batch, A, y))

    if not datasets:
        print("  No batches with enough labels. Skipping tuning.")
        return SCALER_NAME, ALPHA, N_NEIGHBOURS

    param_grid = list(itertools.product(TUNE_SCALERS, TUNE_ALPHAS, TUNE_N_NEIGHBORS))
    total      = len(param_grid)
    records    = []

    for idx, (scaler_name, alpha, k) in enumerate(param_grid):
        f1s, accs, weights = [], [], []
        for batch, A_raw, y in datasets:
            A_norm = normalize_affinity(A_raw, scaler_name)
            f1, acc, _, _ = loo_accuracy(A_norm, y, alpha=alpha, n_neighbors=k,
                                         max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
            if f1 is not None:
                f1s.append(f1)
                accs.append(acc)
                weights.append(int((y != -1).sum()))

        if f1s:
            w        = np.array(weights, dtype=float)
            mean_f1  = float(np.average(f1s,  weights=w))
            mean_acc = float(np.average(accs, weights=w))
        else:
            mean_f1 = mean_acc = float("nan")

        print(f"  [{idx+1}/{total}] scaler={scaler_name}, alpha={alpha}, k={k}"
              f"  ->  macro-F1 = {mean_f1:.3f}  acc = {mean_acc:.3f}")
        records.append({"scaler": scaler_name, "alpha": alpha,
                        "n_neighbors": k, "mean_macro_f1": mean_f1, "mean_acc": mean_acc})

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_DIR / "hyperparam_tuning.csv", index=False)

    # Plot: one heatmap per scaler (alpha vs k)
    for scaler_name in TUNE_SCALERS:
        sub   = results_df[results_df["scaler"] == scaler_name]
        pivot = sub.pivot(index="alpha", columns="n_neighbors", values="mean_macro_f1")
        fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)), max(4, len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                    cbar_kws={"label": "Weighted mean macro-F1"}, ax=ax)
        ax.set_title(f"Tuning — scaler={scaler_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"hyperparam_heatmap_{scaler_name}.png", dpi=150)
        plt.close()

    best      = results_df.loc[results_df["mean_macro_f1"].idxmax()]
    best_scaler = best["scaler"]
    best_alpha  = best["alpha"]
    best_k      = best["n_neighbors"]
    print(f"\n  Best: scaler={best_scaler}, alpha={best_alpha}, k={best_k}"
          f"  (macro-F1={best['mean_macro_f1']:.3f}, acc={best['mean_acc']:.3f})")
    return best_scaler, best_alpha, best_k


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, output_path, title=""):
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                cbar_kws={"label": "Recall"}, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_class_distribution(results_df, output_path, batch_name=""):
    df = results_df.copy()
    df["status"] = "newly assigned"
    df.loc[df["known_label"] != -1, "status"] = "known"
    df.loc[df["final_label"] == UNKNOWN_LABEL, "status"] = "unknown"
    counts = df.groupby(["final_label", "status"]).size().unstack(fill_value=0)
    for col in ["known", "newly assigned", "unknown"]:
        if col not in counts.columns:
            counts[col] = 0
    counts[["known", "newly assigned", "unknown"]].plot(
        kind="bar", stacked=True, figsize=(max(8, len(counts)), 5),
        color=["steelblue", "orange", "lightgrey"])
    plt.xlabel("Condensate class"); plt.ylabel("Number of IDRs")
    plt.title(f"Class distribution — {batch_name}")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(output_path, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_pairs():
    pairs = []
    for aff_path in sorted(DATA_DIR.glob("*.csv")):
        if "test" in aff_path.name:
            continue
        label_path = LABELS_DIR / f"{aff_path.stem}_test.csv"
        if label_path.is_file():
            pairs.append((aff_path, label_path, aff_path.stem))
        else:
            print(f"WARNING: no label file for {aff_path.name}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pairs = find_pairs()
    if not pairs:
        raise RuntimeError("No matching data/label pairs found.")

    scaler_name = SCALER_NAME
    alpha       = ALPHA
    n_neighbors = N_NEIGHBOURS

    if RUN_HYPERPARAMETER_TUNING:
        scaler_name, alpha, n_neighbors = tune_hyperparameters(pairs)
        print(f"\nUsing tuned params: scaler={scaler_name}, alpha={alpha}, k={n_neighbors}")
    else:
        print(f"\nUsing fixed params: scaler={scaler_name}, alpha={alpha}, k={n_neighbors}")

    all_results    = []
    loo_y_true_all = []
    loo_y_pred_all = []

    for aff_path, label_path, batch in pairs:
        print(f"\n{'='*60}\nBatch: {batch}")
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        A = normalize_affinity(A_raw, scaler_name)

        print(f"  IDRs: {len(idr_names)}  |  Labeled: {(y != -1).sum()}"
              f"  |  Scaler: {scaler_name}")

        f1, acc, yt, yp = loo_accuracy(A, y, alpha=alpha, n_neighbors=n_neighbors,
                                        max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
        if f1 is not None:
            print(f"  LOO macro-F1: {f1:.3f}  acc: {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())
            plot_confusion_matrix(yt, yp, OUTPUT_DIR / f"{batch}_confusion.png",
                                  title=f"{batch} — LOO confusion (F1={f1:.2f})")
        else:
            print("  Skipping LOO (too few labeled samples)")

        model, predicted_labels, max_proba = run_label_spreading(
            A, y, alpha=alpha, n_neighbors=n_neighbors,
            max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
        final_labels = derive_final_labels(predicted_labels, max_proba)

        results_df = build_results_df(idr_names, y, predicted_labels, final_labels,
                                      max_proba, model.label_distributions_, model.classes_)
        results_df["batch"] = batch
        all_results.append(results_df)

        plot_class_distribution(results_df, OUTPUT_DIR / f"{batch}_class_dist.png",
                                batch_name=batch)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\nWrote combined results to {OUTPUT_DIR / 'all_results.csv'}")

    if loo_y_true_all:
        plot_confusion_matrix(np.array(loo_y_true_all), np.array(loo_y_pred_all),
                              OUTPUT_DIR / "global_confusion.png",
                              title="Global LOO confusion — normalization")
        gf1  = f1_score(loo_y_true_all, loo_y_pred_all, average="macro", zero_division=0)
        gacc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"Global LOO macro-F1: {gf1:.3f}  acc: {gacc:.3f}")

    plot_class_distribution(combined_df, OUTPUT_DIR / "global_class_dist.png",
                            batch_name="all batches")


if __name__ == "__main__":
    main()
