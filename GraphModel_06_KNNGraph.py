#!/usr/bin/env python3
"""
GraphModel_06_KNNGraph.py

LabelSpreading with an explicit sparse KNN graph built from the affinity
matrix before propagation. Instead of letting LabelSpreading's internal
kernel decide connectivity, we pre-sparsify: keep only the top-k_graph
edges per node (zero out the rest), then symmetrize. This sharpens
neighborhood structure and prevents weak long-range edges from diluting
label propagation.

Tuning loop covers: k_graph (sparsification) x alpha x k_spread (LS kernel).

Usage: python GraphModel_06_KNNGraph.py
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
DATA_DIR   = Path("./data")
LABELS_DIR = Path("./Cleaned_Data")
OUTPUT_DIR = Path("./results_knngraph")

# Fixed hyperparameters (overridden by tuning if enabled)
K_GRAPH      = 20    # edges to keep per node during sparsification
ALPHA        = 0.3   # LabelSpreading clamping factor
N_NEIGHBOURS = 10    # LabelSpreading KNN kernel neighbours (k_spread)
MAX_ITER     = 1000
TOL          = 1e-4
N_JOBS       = -1

CONFIDENCE_THRESHOLD = 0.8
UNKNOWN_LABEL        = 19

LOO_FRACTION = 0.15
LOO_SEED     = 42
LOO_REPEATS  = 10

SKIP_HC_LARGE = False
HC_MAX_SIZE   = 2000

# ---------------------------------------------------------------------------
# Tuning config
# ---------------------------------------------------------------------------
RUN_HYPERPARAMETER_TUNING = False

TUNE_K_GRAPH     = [5, 10, 15, 20]   # sparsification neighbours
TUNE_ALPHAS      = [0.1, 0.3, 0.7, 0.9]
TUNE_N_NEIGHBORS = [10, 20]           # LS kernel neighbours
MIN_LABELS_FOR_TUNING = 10


# ---------------------------------------------------------------------------
# KNN graph sparsification
# ---------------------------------------------------------------------------

def build_knn_graph(A: np.ndarray, k: int) -> np.ndarray:
    """
    Sparsify affinity matrix A by keeping only the top-k neighbours per row.
    The result is symmetrized (union of directed edges) and the diagonal is
    zeroed. Values outside the top-k are set to 0.

    This gives LabelSpreading a sharper graph where each node is only
    connected to its k most similar neighbours, preventing weak affinities
    from polluting propagation.
    """
    n = A.shape[0]
    k = min(k, n - 1)   # can't have more neighbours than nodes - 1
    A_sparse = np.zeros_like(A)

    for i in range(n):
        row = A[i].copy()
        row[i] = -np.inf                          # exclude self
        top_k = np.argpartition(row, -k)[-k:]     # indices of top-k
        A_sparse[i, top_k] = A[i, top_k]

    # Symmetrize: keep edge if either direction has it
    A_sparse = np.maximum(A_sparse, A_sparse.T)
    np.fill_diagonal(A_sparse, 0.0)
    return A_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_affinity_matrix(path: Path):
    df = pd.read_csv(path, index_col=0)
    idr_names = df.index.to_numpy()
    A = df.to_numpy(dtype=float)
    # Negate: FINCHES scores are negative for attraction, positive for repulsion.
    # We flip the sign so that high values = high affinity, as LabelSpreading expects.
    A = -A
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
    print("Hyperparameter tuning — explicit KNN graph sparsification")
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
        return K_GRAPH, ALPHA, N_NEIGHBOURS

    param_grid = list(itertools.product(TUNE_K_GRAPH, TUNE_ALPHAS, TUNE_N_NEIGHBORS))
    total      = len(param_grid)
    records    = []

    for idx, (k_graph, alpha, k_spread) in enumerate(param_grid):
        f1s, accs, weights = [], [], []
        for batch, A_raw, y in datasets:
            # Clamp k_spread to not exceed k_graph (can't spread to more
            # neighbours than the graph has edges)
            k_spread_eff = min(k_spread, k_graph)
            A_sparse = build_knn_graph(A_raw, k_graph)
            f1, acc, _, _ = loo_accuracy(A_sparse, y, alpha=alpha,
                                         n_neighbors=k_spread_eff,
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

        print(f"  [{idx+1}/{total}] k_graph={k_graph}, alpha={alpha}, k_spread={k_spread}"
              f"  ->  macro-F1 = {mean_f1:.3f}  acc = {mean_acc:.3f}")
        records.append({"k_graph": k_graph, "alpha": alpha, "k_spread": k_spread,
                        "mean_macro_f1": mean_f1, "mean_acc": mean_acc})

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_DIR / "hyperparam_tuning.csv", index=False)

    # Heatmap: for each k_graph value, plot alpha vs k_spread
    for kg in TUNE_K_GRAPH:
        sub = results_df[results_df["k_graph"] == kg]
        if sub.empty:
            continue
        try:
            pivot = sub.pivot(index="alpha", columns="k_spread", values="mean_macro_f1")
            fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)), max(4, len(pivot))))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                        cbar_kws={"label": "Weighted mean macro-F1"}, ax=ax)
            ax.set_title(f"Tuning — k_graph={kg}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"hyperparam_heatmap_kgraph{kg}.png", dpi=150)
            plt.close()
        except Exception:
            pass

    best       = results_df.loc[results_df["mean_macro_f1"].idxmax()]
    best_kg    = int(best["k_graph"])
    best_alpha = best["alpha"]
    best_ks    = int(best["k_spread"])
    print(f"\n  Best: k_graph={best_kg}, alpha={best_alpha}, k_spread={best_ks}"
          f"  (macro-F1={best['mean_macro_f1']:.3f}, acc={best['mean_acc']:.3f})")
    return best_kg, best_alpha, best_ks


# ---------------------------------------------------------------------------
# LOO fraction sensitivity
# ---------------------------------------------------------------------------

# Fractions to sweep for the sensitivity analysis
LOO_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]

def loo_fraction_sensitivity(pairs, k_graph, alpha, n_neighbors,
                             fractions=LOO_FRACTIONS):
    """
    For each batch that has enough labels, run LOO at every fraction in
    `fractions` and return a dict:
        {batch: {"fractions": [...], "f1s": [...], "accs": [...], "n_labeled": int}}

    Uses LOO_REPEATS repeats per fraction (same as the main loop) so the
    noise level is consistent.
    """
    print("\n" + "="*60)
    print("LOO fraction sensitivity analysis")
    print("="*60)

    results = {}
    for aff_path, label_path, batch in pairs:
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        n_labeled = int((y != -1).sum())
        if n_labeled < 5:
            print(f"  Skipping {batch} (only {n_labeled} labeled samples)")
            continue

        A = build_knn_graph(A_raw, k_graph)
        k_spread_eff = min(n_neighbors, k_graph)

        batch_f1s, batch_accs = [], []
        for frac in fractions:
            f1, acc, _, _ = loo_accuracy(
                A, y, fraction=frac,
                alpha=alpha, n_neighbors=k_spread_eff,
                max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS,
            )
            batch_f1s.append(f1 if f1 is not None else float("nan"))
            batch_accs.append(acc if acc is not None else float("nan"))
            print(f"  {batch}  frac={frac:.2f}  macro-F1={batch_f1s[-1]:.3f}"
                  f"  acc={batch_accs[-1]:.3f}")

        results[batch] = {
            "fractions": fractions,
            "f1s":       batch_f1s,
            "accs":      batch_accs,
            "n_labeled": n_labeled,
        }

    return results


def plot_loo_sensitivity(sensitivity_results: dict, out_dir: Path):
    """
    One line per batch showing macro-F1 and accuracy as a function of LOO
    fraction. Saved as two figures:
        loo_sensitivity_f1.png
        loo_sensitivity_acc.png

    Batches with very few labels will show noisy/flat lines — that's the
    diagnostic signal: if a line is unstable, the batch is too small to
    trust any single LOO number.
    """
    if not sensitivity_results:
        return

    batches   = list(sensitivity_results.keys())
    fractions = sensitivity_results[batches[0]]["fractions"]
    cmap      = plt.cm.get_cmap("tab10", len(batches))

    for metric_key, metric_label, fname in [
        ("f1s",  "Macro-F1",  "loo_sensitivity_f1.png"),
        ("accs", "Accuracy",  "loo_sensitivity_acc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, batch in enumerate(batches):
            r      = sensitivity_results[batch]
            values = r[metric_key]
            n_lab  = r["n_labeled"]
            ax.plot(
                [f * 100 for f in fractions],
                values,
                marker="o", linewidth=2, markersize=6,
                color=cmap(i),
                label=f"{batch}  (n={n_lab})",
            )

        # Mark the default fraction used in the main run
        ax.axvline(LOO_FRACTION * 100, color="red", linestyle="--",
                   linewidth=1.2, label=f"Default ({int(LOO_FRACTION*100)}%)")

        ax.set_xlabel("LOO holdout fraction (%)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(
            f"LOO {metric_label} vs holdout fraction — KNN graph\n"
            f"(k_graph={K_GRAPH}, alpha={ALPHA}, k_spread={N_NEIGHBOURS}, "
            f"{LOO_REPEATS} repeats)",
            fontsize=11,
        )
        ax.set_xticks([f * 100 for f in fractions])
        ax.set_xticklabels([f"{int(f*100)}%" for f in fractions])
        y_vals = [v for r in sensitivity_results.values()
                  for v in r[metric_key] if not np.isnan(v)]
        if y_vals:
            ax.set_ylim(max(0.0, min(y_vals) - 0.05),
                        min(1.0, max(y_vals) + 0.08))
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Accuracy summary plots
# ---------------------------------------------------------------------------

def plot_accuracy_summary(batch_loo: dict, global_f1: float, global_acc: float,
                          out_dir: Path):
    """
    Two grouped bar charts (macro-F1 and accuracy) — one bar per batch plus
    a global bar — built from the LOO scores collected during the main loop.
    Batches are ordered from largest to smallest by n_labeled, then GLOBAL.

    batch_loo: {batch: {"f1": float, "acc": float, "n_labeled": int}}
    """
    # Sort batches by n_labeled descending
    batches  = sorted(batch_loo.keys(), key=lambda b: batch_loo[b]["n_labeled"], reverse=True)
    f1s      = [batch_loo[b]["f1"]  for b in batches]
    accs     = [batch_loo[b]["acc"] for b in batches]
    n_labels = [batch_loo[b]["n_labeled"] for b in batches]

    # Append global
    all_labels = batches + ["GLOBAL"]
    all_f1s    = f1s  + [global_f1]
    all_accs   = accs + [global_acc]
    all_n      = n_labels + [sum(n_labels)]

    x     = np.arange(len(all_labels))
    width = 0.35
    colors_batch  = ["steelblue"] * len(batches) + ["#2c7bb6"]
    colors_global = ["orange"]    * len(batches) + ["#d7191c"]

    fig, ax = plt.subplots(figsize=(max(12, len(all_labels) * 1.3), 5))
    bars1 = ax.bar(x - width/2, all_f1s,  width, label="Macro-F1",
                   color=colors_batch, alpha=0.85)
    bars2 = ax.bar(x + width/2, all_accs, width, label="Accuracy",
                   color=colors_global, alpha=0.85)

    # Annotate with n_labeled
    for i, (bar, n) in enumerate(zip(bars1, all_n)):
        ax.text(bar.get_x() + bar.get_width()/2,
                -0.04, f"n={n}", ha="center", va="top",
                fontsize=7, color="dimgrey", rotation=45)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    # Shade the global bar region
    ax.axvspan(len(batches) - 0.5, len(all_labels) - 0.5,
               alpha=0.08, color="grey", label="Global")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, min(1.0, max(filter(lambda v: not np.isnan(v),
                                       all_f1s + all_accs)) + 0.15))
    ax.set_title(
        f"LOO evaluation per batch — KNN graph\n"
        f"(k_graph={K_GRAPH}, alpha={ALPHA}, k_spread={N_NEIGHBOURS}, "
        f"{LOO_REPEATS} repeats × {int(LOO_FRACTION*100)}% holdout)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_summary.png", dpi=150)
    plt.close()
    print("  Saved accuracy_summary.png")


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


def plot_hc_vs_ls(A, results_df, output_path, batch_name=""):
    n = len(A)
    if SKIP_HC_LARGE and n > HC_MAX_SIZE:
        print(f"  Skipping HC for {batch_name} (n={n})")
        return None
    # Convert affinity to distance: shift so min=0, then invert
    A_shifted = A - A.min()
    A_norm = A_shifted / A_shifted.max() if A_shifted.max() > 0 else A_shifted
    D = 1.0 - A_norm
    np.fill_diagonal(D, 0.0)
    n_clusters = len(results_df["final_label"].unique())
    from sklearn.cluster import AgglomerativeClustering
    hc_labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(D)
    ls_labels = results_df["final_label"].to_numpy()
    ari = adjusted_rand_score(hc_labels, ls_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(hc_labels, ls_labels, c=results_df["max_proba"].to_numpy(),
                    cmap="viridis", alpha=0.4, s=10)
    plt.colorbar(sc, ax=ax, label="LS confidence")
    ax.set_xlabel("HC label"); ax.set_ylabel("LS label")
    ax.set_title(f"{batch_name} — HC vs LS (ARI={ari:.3f})")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    return ari


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

    k_graph     = K_GRAPH
    alpha       = ALPHA
    n_neighbors = N_NEIGHBOURS

    if RUN_HYPERPARAMETER_TUNING:
        k_graph, alpha, n_neighbors = tune_hyperparameters(pairs)
        print(f"\nUsing tuned params: k_graph={k_graph}, alpha={alpha}, k_spread={n_neighbors}")
    else:
        print(f"\nUsing fixed params: k_graph={k_graph}, alpha={alpha}, k_spread={n_neighbors}")

    all_results    = []
    loo_y_true_all = []
    loo_y_pred_all = []
    ari_scores     = {}
    batch_loo      = {}   # {batch: {"f1": float, "acc": float, "n_labeled": int}}

    for aff_path, label_path, batch in pairs:
        print(f"\n{'='*60}\nBatch: {batch}")
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        A = build_knn_graph(A_raw, k_graph)

        n_labeled = (y != -1).sum()
        print(f"  IDRs: {len(idr_names)}  |  Labeled: {n_labeled}"
              f"  |  k_graph: {k_graph}")

        k_spread_eff = min(n_neighbors, k_graph)
        f1, acc, yt, yp = loo_accuracy(A, y, alpha=alpha, n_neighbors=k_spread_eff,
                                        max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
        if f1 is not None:
            print(f"  LOO macro-F1: {f1:.3f}  acc: {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())
            batch_loo[batch] = {"f1": f1, "acc": acc, "n_labeled": int(n_labeled)}
            plot_confusion_matrix(yt, yp, OUTPUT_DIR / f"{batch}_confusion.png",
                                  title=f"{batch} — LOO confusion (F1={f1:.2f})")
        else:
            print("  Skipping LOO (too few labeled samples)")

        model, predicted_labels, max_proba = run_label_spreading(
            A, y, alpha=alpha, n_neighbors=k_spread_eff,
            max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS)
        final_labels = derive_final_labels(predicted_labels, max_proba)

        results_df = build_results_df(idr_names, y, predicted_labels, final_labels,
                                      max_proba, model.label_distributions_, model.classes_)
        results_df["batch"] = batch
        all_results.append(results_df)

        plot_class_distribution(results_df, OUTPUT_DIR / f"{batch}_class_dist.png",
                                batch_name=batch)

        ari = plot_hc_vs_ls(A, results_df, OUTPUT_DIR / f"{batch}_hc_vs_ls.png",
                            batch_name=batch)
        ari_scores[batch] = ari
        if ari is not None:
            print(f"  ARI (HC vs LS): {ari:.3f}")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\nWrote combined results to {OUTPUT_DIR / 'all_results.csv'}")

    if loo_y_true_all:
        plot_confusion_matrix(np.array(loo_y_true_all), np.array(loo_y_pred_all),
                              OUTPUT_DIR / "global_confusion.png",
                              title="Global LOO confusion — KNN graph")
        gf1  = f1_score(loo_y_true_all, loo_y_pred_all, average="macro", zero_division=0)
        gacc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"Global LOO macro-F1: {gf1:.3f}  acc: {gacc:.3f}")
        print("\n  Generating accuracy summary plot...")
        plot_accuracy_summary(batch_loo, float(gf1), float(gacc), OUTPUT_DIR)

    print("\nARI scores:")
    for batch, ari in ari_scores.items():
        print(f"  {batch}: {ari:.3f}" if ari is not None else f"  {batch}: skipped")

    plot_class_distribution(combined_df, OUTPUT_DIR / "global_class_dist.png",
                            batch_name="all batches")

    # --- LOO fraction sensitivity ---
    print("\n  Running LOO fraction sensitivity analysis...")
    sensitivity = loo_fraction_sensitivity(pairs, k_graph, alpha,
                                           min(n_neighbors, k_graph))
    plot_loo_sensitivity(sensitivity, OUTPUT_DIR)


if __name__ == "__main__":
    main()
