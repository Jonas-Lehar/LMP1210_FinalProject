#!/usr/bin/env python3
"""
GraphModel_09_Final.py

Final model run using corrected FINCHES preprocessing and best hyperparameters
from GM08 grid search.

Preprocessing pipeline:
  1. Sign flip   — negative attraction -> positive similarity
  2. Clip        — repulsive pairs (still negative) -> 0
  3. MinMax      — scale to [0, 1]

Best hyperparameters (from GM08):
  row_norm       = none
  k_graph        = 25
  alpha          = 0.5
  conf_threshold = 0.8

Runs on ALL batches. Produces:
  - all_results.csv
  - Per-batch: confusion matrix, class distribution, HC vs LS, confidence
  - Global: confusion matrix, class distribution, accuracy summary
  - LOO fraction sensitivity curves

Usage: python GraphModel_09_Final.py
"""

from pathlib import Path
import sys

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
OUTPUT_DIR = Path("./results_final")

# Best hyperparameters from GM08
K_GRAPH          = 25
ALPHA            = 0.5
N_NEIGHBOURS     = 20    # k_spread = min(K_GRAPH, 20) = 20
CONF_THRESHOLD   = 0.8
UNKNOWN_LABEL    = 19

MAX_ITER = 1000
TOL      = 1e-4
N_JOBS   = -1

LOO_FRACTION = 0.15
LOO_SEED     = 42
LOO_REPEATS  = 10

# HC skipped for matrices larger than this
HC_MAX_SIZE = 5000

# LOO sensitivity fractions
LOO_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30]

# Canonical batch order — ascending IDR length
BATCH_ORDER = ["50_99", "100_150", "151_200", "201_250",
               "251_300", "301_350", "351_400", "401_450"]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_affinity(A: np.ndarray) -> np.ndarray:
    """
    Correct preprocessing for FINCHES affinity matrices:
      1. Sign flip  — negative ε (attraction) becomes positive similarity
      2. Clip       — repulsive pairs (positive ε, now negative) -> 0
      3. MinMax     — scale to [0, 1]
    """
    A = -A.copy()
    A = np.clip(A, 0.0, None)
    np.fill_diagonal(A, 0.0)
    vmax = A.max()
    if vmax > 0:
        A = A / vmax
    np.fill_diagonal(A, 0.0)
    return A


def build_knn_graph(A: np.ndarray, k: int) -> np.ndarray:
    n = A.shape[0]
    k = min(k, n - 1)
    A_sparse = np.zeros_like(A)
    for i in range(n):
        row = A[i].copy()
        row[i] = -np.inf
        top_k = np.argpartition(row, -k)[-k:]
        A_sparse[i, top_k] = A[i, top_k]
    A_sparse = np.maximum(A_sparse, A_sparse.T)
    np.fill_diagonal(A_sparse, 0.0)
    return A_sparse


# ---------------------------------------------------------------------------
# Data loading
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


def find_pairs():
    """Find all batch pairs, returned in ascending IDR length order."""
    available = {}
    for aff_path in DATA_DIR.glob("*.csv"):
        if "test" in aff_path.name:
            continue
        label_path = LABELS_DIR / f"{aff_path.stem}_test.csv"
        if label_path.is_file():
            available[aff_path.stem] = (aff_path, label_path)
        else:
            print(f"WARNING: no label file for {aff_path.name}", file=sys.stderr)

    # Return in canonical ascending order, skip any missing
    pairs = []
    for batch in BATCH_ORDER:
        if batch in available:
            aff_path, label_path = available[batch]
            pairs.append((aff_path, label_path, batch))
        else:
            print(f"WARNING: batch {batch} not found in {DATA_DIR}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# LabelSpreading
# ---------------------------------------------------------------------------

def run_label_spreading(A, y, alpha=ALPHA, n_neighbors=N_NEIGHBOURS):
    model = LabelSpreading(
        kernel="knn", n_neighbors=int(n_neighbors),
        alpha=alpha, max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS,
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
    final_labels[max_proba < CONF_THRESHOLD] = UNKNOWN_LABEL
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
                 n_repeats=LOO_REPEATS, alpha=ALPHA, n_neighbors=N_NEIGHBOURS):
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
        _, pred, _ = run_label_spreading(A, y_masked, alpha, n_neighbors)
        y_true = y[hold_idx]
        y_pred = pred[hold_idx]
        f1s.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        accs.append((y_true == y_pred).mean())
        y_true_all.extend(y_true.tolist())
        y_pred_all.extend(y_pred.tolist())
    return float(np.mean(f1s)), float(np.mean(accs)), \
           np.array(y_true_all), np.array(y_pred_all)


def loo_fraction_sensitivity(pairs):
    """
    Run LOO at multiple holdout fractions for each batch with enough labels.
    Returns {batch: {"fractions": [...], "f1s": [...], "accs": [...], "n_labeled": int}}
    """
    print("\n" + "="*60)
    print("LOO fraction sensitivity analysis")
    print("="*60)
    results = {}
    k_spread = min(N_NEIGHBOURS, K_GRAPH)

    for aff_path, label_path, batch in pairs:
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        n_labeled = int((y != -1).sum())
        if n_labeled < 5:
            print(f"  Skipping {batch} (only {n_labeled} labeled samples)")
            continue
        A = build_knn_graph(preprocess_affinity(A_raw), K_GRAPH)
        batch_f1s, batch_accs = [], []
        for frac in LOO_FRACTIONS:
            f1, acc, _, _ = loo_accuracy(A, y, fraction=frac,
                                         alpha=ALPHA, n_neighbors=k_spread)
            batch_f1s.append(f1 if f1 is not None else float("nan"))
            batch_accs.append(acc if acc is not None else float("nan"))
            print(f"  {batch}  frac={frac:.2f}  F1={batch_f1s[-1]:.3f}"
                  f"  acc={batch_accs[-1]:.3f}")
        results[batch] = {"fractions": LOO_FRACTIONS, "f1s": batch_f1s,
                          "accs": batch_accs, "n_labeled": n_labeled}
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _batch_sort_key(batch):
    """Sort key that orders batches by ascending IDR length range."""
    try:
        return BATCH_ORDER.index(batch)
    except ValueError:
        return 999


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


def plot_confidence_by_class(results_df, output_path, title=""):
    df = results_df[results_df["final_label"] != UNKNOWN_LABEL].copy()
    if df.empty:
        return
    order = sorted(df["final_label"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(order)), 5))
    sns.violinplot(data=df, x="final_label", y="max_proba", hue="final_label",
                   order=order, inner="quartile", palette="muted", legend=False, ax=ax)
    ax.axhline(CONF_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({CONF_THRESHOLD})")
    ax.set_xlabel("Condensate class"); ax.set_ylabel("Max probability")
    ax.set_title(title or "Confidence by class")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_hc_vs_ls(A, results_df, output_path, batch_name=""):
    n = len(A)
    if n > HC_MAX_SIZE:
        print(f"  Skipping HC for {batch_name} (n={n}, too large)")
        return None
    D = 1.0 - np.clip(A, 0, 1)
    np.fill_diagonal(D, 0.0)
    n_clusters = len(results_df["final_label"].unique())
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


def plot_accuracy_summary(batch_loo: dict, global_f1: float, global_acc: float,
                          out_dir: Path):
    """Grouped bar chart of macro-F1 and accuracy per batch + global."""
    batches    = sorted(batch_loo.keys(), key=_batch_sort_key)
    all_labels = batches + ["GLOBAL"]
    all_f1s    = [batch_loo[b]["f1"]  for b in batches] + [global_f1]
    all_accs   = [batch_loo[b]["acc"] for b in batches] + [global_acc]
    all_n      = [batch_loo[b]["n_labeled"] for b in batches] + \
                 [sum(batch_loo[b]["n_labeled"] for b in batches)]

    x = np.arange(len(all_labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(all_labels) * 1.3), 5))
    bars1 = ax.bar(x - width/2, all_f1s,  width, label="Macro-F1",
                   color=["steelblue"]*len(batches) + ["#2c7bb6"], alpha=0.85)
    bars2 = ax.bar(x + width/2, all_accs, width, label="Accuracy",
                   color=["orange"]*len(batches) + ["#d7191c"], alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axvspan(len(batches) - 0.5, len(all_labels) - 0.5,
               alpha=0.08, color="grey", label="Global")
    ax.set_xticks(x); ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, min(1.0, max(v for v in all_f1s + all_accs
                                if not np.isnan(v)) + 0.15))
    ax.set_title(
        f"LOO evaluation per batch — GM09 Final\n"
        f"(k_graph={K_GRAPH}, alpha={ALPHA}, k_spread={min(N_NEIGHBOURS, K_GRAPH)}, "
        f"conf={CONF_THRESHOLD}, {LOO_REPEATS} repeats × {int(LOO_FRACTION*100)}% holdout)",
        fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_summary.png", dpi=150); plt.close()
    print("  Saved accuracy_summary.png")


def plot_loo_sensitivity(sensitivity_results: dict, out_dir: Path):
    """Line plot of macro-F1 and accuracy vs LOO fraction, one line per batch."""
    if not sensitivity_results:
        return
    batches   = sorted(sensitivity_results.keys(), key=_batch_sort_key)
    fractions = sensitivity_results[batches[0]]["fractions"]
    cmap      = matplotlib.colormaps.get_cmap("tab10")

    for metric_key, metric_label, fname in [
        ("f1s",  "Macro-F1", "loo_sensitivity_f1.png"),
        ("accs", "Accuracy", "loo_sensitivity_acc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, batch in enumerate(batches):
            r = sensitivity_results[batch]
            ax.plot([f * 100 for f in fractions], r[metric_key],
                    marker="o", linewidth=2, markersize=6,
                    color=cmap(i), label=f"{batch}  (n={r['n_labeled']})")
        ax.axvline(LOO_FRACTION * 100, color="red", linestyle="--",
                   linewidth=1.2, label=f"Default ({int(LOO_FRACTION*100)}%)")
        ax.set_xlabel("LOO holdout fraction (%)", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(
            f"LOO {metric_label} vs holdout fraction — GM09 Final\n"
            f"(k_graph={K_GRAPH}, alpha={ALPHA}, {LOO_REPEATS} repeats)", fontsize=11)
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
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved {fname}")


def plot_ari_summary(ari_scores: dict, out_dir: Path):
    """Bar chart of ARI per batch in ascending order."""
    batches = sorted(ari_scores.keys(), key=_batch_sort_key)
    values  = [ari_scores[b] for b in batches if ari_scores[b] is not None]
    labels  = [b for b in batches if ari_scores[b] is not None]
    if not values:
        return
    colors = ["tomato" if v < 0 else "steelblue" for v in values]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if v >= 0 else -0.008),
                f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_xlabel("Batch (IDR length range)", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title("ARI: Hierarchical Clustering vs LabelSpreading per batch", fontsize=12)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ari_summary.png", dpi=150); plt.close()
    print("  Saved ari_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    pairs = find_pairs()
    if not pairs:
        raise RuntimeError("No matching data/label pairs found.")

    k_spread = min(N_NEIGHBOURS, K_GRAPH)
    print(f"GM09 Final — k_graph={K_GRAPH}, alpha={ALPHA}, "
          f"k_spread={k_spread}, conf={CONF_THRESHOLD}")

    # Sanity check preprocessing on first batch
    idr_names, A_raw = load_affinity_matrix(pairs[0][0])
    A_check = preprocess_affinity(A_raw)
    print(f"\nPreprocessing check on {pairs[0][2]}:")
    print(f"  Raw range:  [{A_raw.min():.3f}, {A_raw.max():.3f}]")
    print(f"  After prep: [{A_check.min():.3f}, {A_check.max():.3f}]")
    print(f"  % zeros:    {(A_check == 0).mean()*100:.1f}%")
    print(f"  % negative: {(A_check < 0).mean()*100:.1f}%  (should be 0)")

    all_results    = []
    loo_y_true_all = []
    loo_y_pred_all = []
    ari_scores     = {}
    batch_loo      = {}

    for aff_path, label_path, batch in pairs:
        print(f"\n{'='*60}\nBatch: {batch}")
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y        = load_known_labels(label_path, idr_names)
        A_proc   = preprocess_affinity(A_raw)
        A        = build_knn_graph(A_proc, K_GRAPH)
        n_labeled = int((y != -1).sum())
        print(f"  IDRs: {len(idr_names)}  |  Labeled: {n_labeled}")

        # LOO evaluation
        f1, acc, yt, yp = loo_accuracy(A, y, alpha=ALPHA, n_neighbors=k_spread)
        if f1 is not None:
            print(f"  LOO macro-F1: {f1:.3f}  acc: {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())
            batch_loo[batch] = {"f1": f1, "acc": acc, "n_labeled": n_labeled}
            plot_confusion_matrix(yt, yp, OUTPUT_DIR / f"{batch}_confusion.png",
                                  title=f"{batch} — LOO confusion (F1={f1:.2f})")
        else:
            print("  Skipping LOO (too few labeled samples)")

        # Full propagation
        model, pred_labels, max_proba = run_label_spreading(A, y, ALPHA, k_spread)
        final_labels = derive_final_labels(pred_labels, max_proba)
        n_assigned = max(0, int((final_labels != UNKNOWN_LABEL).sum()) - n_labeled)
        mean_conf  = float(max_proba[final_labels != UNKNOWN_LABEL].mean())
        print(f"  Newly assigned: {n_assigned}  |  Mean confidence: {mean_conf:.3f}")

        results_df = build_results_df(idr_names, y, pred_labels, final_labels,
                                      max_proba, model.label_distributions_, model.classes_)
        results_df["batch"] = batch
        all_results.append(results_df)

        plot_class_distribution(results_df, OUTPUT_DIR / f"{batch}_class_dist.png",
                                batch_name=batch)
        plot_confidence_by_class(results_df, OUTPUT_DIR / f"{batch}_confidence.png",
                                 title=f"{batch} — confidence by class")
        ari = plot_hc_vs_ls(A, results_df, OUTPUT_DIR / f"{batch}_hc_vs_ls.png",
                            batch_name=batch)
        ari_scores[batch] = ari
        if ari is not None:
            print(f"  ARI (HC vs LS): {ari:.3f}")

    # Save combined results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\nWrote combined results to {OUTPUT_DIR / 'all_results.csv'}")

    # Global confusion matrix + accuracy summary
    if loo_y_true_all:
        plot_confusion_matrix(np.array(loo_y_true_all), np.array(loo_y_pred_all),
                              OUTPUT_DIR / "global_confusion.png",
                              title="Global LOO confusion — GM09 Final")
        gf1  = f1_score(loo_y_true_all, loo_y_pred_all, average="macro", zero_division=0)
        gacc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"Global LOO macro-F1: {gf1:.3f}  acc: {gacc:.3f}")
        plot_accuracy_summary(batch_loo, float(gf1), float(gacc), OUTPUT_DIR)

    # Global class distribution
    plot_class_distribution(combined_df, OUTPUT_DIR / "global_class_dist.png",
                            batch_name="all batches")

    # ARI summary
    print("\nARI scores:")
    for batch in sorted(ari_scores.keys(), key=_batch_sort_key):
        ari = ari_scores[batch]
        print(f"  {batch}: {ari:.3f}" if ari is not None else f"  {batch}: skipped")
    plot_ari_summary(ari_scores, OUTPUT_DIR)

    # LOO fraction sensitivity
    print("\nRunning LOO fraction sensitivity analysis...")
    sensitivity = loo_fraction_sensitivity(pairs)
    plot_loo_sensitivity(sensitivity, OUTPUT_DIR)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("\nTo generate post-run analysis figures run:")
    print(f"  python analysis_results.py --results {OUTPUT_DIR}/all_results.csv "
          f"--output results_analysis_final")


if __name__ == "__main__":
    main()
