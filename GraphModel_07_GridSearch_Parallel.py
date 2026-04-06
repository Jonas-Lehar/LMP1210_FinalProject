#!/usr/bin/env python3
"""
GraphModel_07_GridSearch_Parallel.py

Parallel version of GraphModel_07_GridSearch.py.
Uses joblib.Parallel to run grid search combos concurrently across all
available CPU cores. Everything else is identical.

Speedup scales roughly linearly with core count up to ~8-12 cores.
Note: N_JOBS inside LabelSpreading is set to 1 here to avoid nested
parallelism fighting over cores (outer parallel loop already uses them all).

Usage: python GraphModel_07_GridSearch_Parallel.py
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
from joblib import Parallel, delayed
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = Path("./data")
LABELS_DIR = Path("./Cleaned_Data")
OUTPUT_DIR = Path("./results_gridsearch_parallel")

PRIORITY_BATCHES = {"50_99", "100_150", "151_200"}

MAX_ITER = 1000
TOL      = 1e-4
# N_JOBS for the outer parallel loop (uses all cores)
N_PARALLEL_JOBS = -1
# N_JOBS inside LabelSpreading — keep at 1 to avoid nested parallelism
LS_N_JOBS = 1

UNKNOWN_LABEL = 19

LOO_FRACTION = 0.15
LOO_SEED     = 42
LOO_REPEATS  = 10

# ---------------------------------------------------------------------------
# Grid search space
# ---------------------------------------------------------------------------
TUNE_NORMALIZATIONS  = ["none", "minmax", "standard", "robust"]
TUNE_K_GRAPH         = [10, 15, 20, 30, 50]
TUNE_ALPHAS          = [0.1, 0.2, 0.3, 0.5, 0.7]
TUNE_CONF_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

DEFAULT_NORM   = "none"
DEFAULT_K      = 15
DEFAULT_ALPHA  = 0.3
DEFAULT_CONF   = 0.6


# ---------------------------------------------------------------------------
# Normalization / graph / data loading  (identical to non-parallel version)
# ---------------------------------------------------------------------------

def normalize_affinity(A: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return A
    scalers = {
        "minmax":   MinMaxScaler(),
        "standard": StandardScaler(),
        "robust":   RobustScaler(),
    }
    if method not in scalers:
        raise ValueError(f"Unknown normalization '{method}'")
    A_norm = scalers[method].fit_transform(A)
    np.fill_diagonal(A_norm, 0.0)
    return A_norm


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


def load_affinity_matrix(path: Path):
    df = pd.read_csv(path, index_col=0)
    idr_names = df.index.to_numpy()
    A = df.to_numpy(dtype=float)
    A = -A   # sign flip: FINCHES negative = attraction
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
    seen, pairs = set(), []
    for aff_path in sorted(DATA_DIR.glob("**/*.csv")):
        if "test" in aff_path.name:
            continue
        if aff_path.stem not in PRIORITY_BATCHES:
            continue
        if aff_path.stem in seen:
            continue
        seen.add(aff_path.stem)
        label_path = LABELS_DIR / f"{aff_path.stem}_test.csv"
        if label_path.is_file():
            pairs.append((aff_path, label_path, aff_path.stem))
        else:
            print(f"WARNING: no label file for {aff_path.name}", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# LabelSpreading (LS_N_JOBS=1 to avoid nested parallelism)
# ---------------------------------------------------------------------------

def run_label_spreading(A, y, alpha, n_neighbors):
    model = LabelSpreading(
        kernel="knn", n_neighbors=int(n_neighbors),
        alpha=alpha, max_iter=MAX_ITER, tol=TOL, n_jobs=LS_N_JOBS,
    )
    model.fit(A, y)
    proba            = model.label_distributions_
    classes          = model.classes_
    argmax_idx       = proba.argmax(axis=1)
    max_proba        = proba[np.arange(len(proba)), argmax_idx]
    predicted_labels = classes[argmax_idx]
    return model, predicted_labels, max_proba


def derive_final_labels(predicted_labels, max_proba, confidence_threshold):
    final_labels = predicted_labels.copy()
    final_labels[max_proba < confidence_threshold] = UNKNOWN_LABEL
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

def loo_accuracy(A, y, alpha, n_neighbors,
                 fraction=LOO_FRACTION, seed=LOO_SEED, n_repeats=LOO_REPEATS):
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


# ---------------------------------------------------------------------------
# Single combo worker  — this is what gets parallelized
# ---------------------------------------------------------------------------

def _evaluate_combo(norm, k_graph, alpha, datasets):
    """
    Evaluate one (norm, k_graph, alpha) combination across all datasets.
    Returns a list of records (one per conf_threshold).
    Called by joblib workers — must be a top-level function (picklable).
    """
    f1s, accs, weights = [], [], []
    batch_probas = {}

    for batch, _idr_names, A_raw, y in datasets:
        A_norm   = normalize_affinity(A_raw, norm)
        A_sparse = build_knn_graph(A_norm, k_graph)
        k_spread = min(k_graph, 20)

        f1, acc, _, _ = loo_accuracy(A_sparse, y, alpha, k_spread)
        if f1 is not None:
            f1s.append(f1)
            accs.append(acc)
            weights.append(int((y != -1).sum()))

        _, _pred, max_proba = run_label_spreading(A_sparse, y, alpha, k_spread)
        batch_probas[batch] = (y, max_proba)

    if f1s:
        w        = np.array(weights, dtype=float)
        mean_f1  = float(np.average(f1s,  weights=w))
        mean_acc = float(np.average(accs, weights=w))
    else:
        mean_f1 = mean_acc = float("nan")

    records = []
    for conf in TUNE_CONF_THRESHOLDS:
        conf_vals, pct_vals = [], []
        for b_name, _idr_names, _A_raw, y_b in datasets:
            y_batch, max_proba = batch_probas[b_name]
            unlabeled_mask = y_b == -1
            if unlabeled_mask.sum() == 0:
                continue
            assigned_mask = unlabeled_mask & (max_proba >= conf)
            pct       = float(assigned_mask.sum() / unlabeled_mask.sum())
            mean_conf = float(max_proba[assigned_mask].mean()) if assigned_mask.any() else 0.0
            conf_vals.append(mean_conf)
            pct_vals.append(pct)

        records.append({
            "norm":               norm,
            "k_graph":            k_graph,
            "alpha":              alpha,
            "conf_threshold":     conf,
            "mean_macro_f1":      mean_f1,
            "mean_acc":           mean_acc,
            "mean_conf_assigned": float(np.mean(conf_vals)) if conf_vals else float("nan"),
            "pct_assigned":       float(np.mean(pct_vals))  if pct_vals  else float("nan"),
        })
    return records


# ---------------------------------------------------------------------------
# Parallel grid search
# ---------------------------------------------------------------------------

def run_grid_search(pairs):
    print("\n" + "="*60)
    print("Parallel grid search — normalization x k_graph x alpha x conf_threshold")
    print(f"Batches: {[b for _,_,b in pairs]}")
    print("="*60)

    # Pre-load all datasets into memory (shared across workers via joblib)
    datasets = []
    for aff_path, label_path, batch in pairs:
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        n_labeled = int((y != -1).sum())
        print(f"  Loaded {batch}: {len(idr_names)} IDRs, {n_labeled} labeled")
        datasets.append((batch, idr_names, A_raw, y))

    prop_grid  = list(itertools.product(TUNE_NORMALIZATIONS, TUNE_K_GRAPH, TUNE_ALPHAS))
    total_prop = len(prop_grid)
    print(f"\nRunning {total_prop} combos in parallel (N_PARALLEL_JOBS={N_PARALLEL_JOBS})...")

    # joblib dispatches each combo to a worker process
    # prefer="threads" avoids pickling large numpy arrays across processes
    nested_records = Parallel(n_jobs=N_PARALLEL_JOBS, prefer="threads", verbose=5)(
        delayed(_evaluate_combo)(norm, k_graph, alpha, datasets)
        for norm, k_graph, alpha in prop_grid
    )

    # Flatten list-of-lists
    all_records = [rec for combo_recs in nested_records for rec in combo_recs]

    results_df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "grid_search_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved full grid results to {csv_path}")
    return results_df


# ---------------------------------------------------------------------------
# Everything below is identical to the non-parallel version
# ---------------------------------------------------------------------------

def pick_best_params(results_df: pd.DataFrame):
    ranked = results_df.sort_values(
        ["mean_macro_f1", "mean_conf_assigned"], ascending=[False, False]
    )
    best = ranked.iloc[0]
    print("\n" + "="*60)
    print("Best hyperparameters")
    print("="*60)
    print(f"  norm           = {best['norm']}")
    print(f"  k_graph        = {int(best['k_graph'])}")
    print(f"  alpha          = {best['alpha']}")
    print(f"  conf_threshold = {best['conf_threshold']}")
    print(f"  macro-F1       = {best['mean_macro_f1']:.3f}")
    print(f"  accuracy       = {best['mean_acc']:.3f}")
    print(f"  mean conf (assigned) = {best['mean_conf_assigned']:.3f}")
    print(f"  % unlabeled assigned = {best['pct_assigned']*100:.1f}%")
    return (best["norm"], int(best["k_graph"]),
            float(best["alpha"]), float(best["conf_threshold"]))


def plot_grid_heatmaps(results_df: pd.DataFrame, out_dir: Path):
    for norm in TUNE_NORMALIZATIONS:
        sub = results_df[results_df["norm"] == norm]
        agg = sub.groupby(["alpha", "k_graph"])[["mean_macro_f1", "mean_conf_assigned"]].mean().reset_index()
        for metric, label, fname in [
            ("mean_macro_f1",      "Weighted mean macro-F1",        f"heatmap_{norm}_f1.png"),
            ("mean_conf_assigned", "Mean confidence of assignments", f"heatmap_{norm}_conf.png"),
        ]:
            try:
                pivot = agg.pivot(index="alpha", columns="k_graph", values=metric)
                fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)), max(4, len(pivot))))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu",
                            cbar_kws={"label": label}, ax=ax)
                ax.set_title(f"norm={norm} — {label}\n(averaged over conf_threshold)")
                ax.set_xlabel("k_graph"); ax.set_ylabel("alpha")
                plt.tight_layout()
                plt.savefig(out_dir / fname, dpi=150); plt.close()
            except Exception as e:
                print(f"  Warning: could not plot {fname}: {e}")

    best_prop  = results_df.loc[results_df["mean_macro_f1"].idxmax()]
    best_norm  = best_prop["norm"]
    best_k     = best_prop["k_graph"]
    best_alpha = best_prop["alpha"]
    sub = results_df[
        (results_df["norm"]    == best_norm) &
        (results_df["k_graph"] == best_k)    &
        (results_df["alpha"]   == best_alpha)
    ].sort_values("conf_threshold")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(sub["conf_threshold"], sub["mean_macro_f1"],
             marker="o", color="steelblue", label="Macro-F1")
    ax2.plot(sub["conf_threshold"], sub["pct_assigned"] * 100,
             marker="s", color="orange", linestyle="--", label="% assigned")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Macro-F1", color="steelblue")
    ax2.set_ylabel("% unlabeled IDRs assigned", color="orange")
    ax1.set_title(f"F1 vs confidence threshold\n"
                  f"(norm={best_norm}, k={int(best_k)}, alpha={best_alpha})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "conf_threshold_tradeoff.png", dpi=150); plt.close()
    print("  Saved conf_threshold_tradeoff.png")


def plot_normalization_comparison(results_df: pd.DataFrame, out_dir: Path):
    best_per_norm = (results_df.groupby("norm")["mean_macro_f1"]
                     .max().reset_index()
                     .sort_values("mean_macro_f1", ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(best_per_norm["norm"], best_per_norm["mean_macro_f1"],
                  color="steelblue", alpha=0.85)
    for bar, v in zip(bars, best_per_norm["mean_macro_f1"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Normalization method")
    ax.set_ylabel("Best weighted macro-F1")
    ax.set_title("Best LOO macro-F1 per normalization method")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "normalization_comparison.png", dpi=150); plt.close()
    print("  Saved normalization_comparison.png")


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


def plot_confidence_by_class(results_df, output_path, conf_threshold=0.6, title=""):
    df = results_df[results_df["final_label"] != UNKNOWN_LABEL].copy()
    if df.empty:
        return
    order = sorted(df["final_label"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(order)), 5))
    sns.violinplot(data=df, x="final_label", y="max_proba", hue="final_label",
                   order=order, inner="quartile", palette="muted", legend=False, ax=ax)
    ax.axhline(conf_threshold, color="red", linestyle="--",
               label=f"Threshold ({conf_threshold})")
    ax.set_xlabel("Condensate class"); ax.set_ylabel("Max probability")
    ax.set_title(title or "Confidence by class"); ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_hc_vs_ls(A, results_df, output_path, batch_name=""):
    n = len(A)
    if n > 5000:
        print(f"  Skipping HC for {batch_name} (n={n}, too large)")
        return None
    A_shifted = A - A.min()
    denom  = A_shifted.max()
    A_norm = A_shifted / denom if denom > 0 else A_shifted
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


def plot_accuracy_summary(batch_loo, global_f1, global_acc, out_dir):
    batches    = sorted(batch_loo.keys(), key=lambda b: batch_loo[b]["n_labeled"], reverse=True)
    all_labels = batches + ["GLOBAL"]
    all_f1s    = [batch_loo[b]["f1"]  for b in batches] + [global_f1]
    all_accs   = [batch_loo[b]["acc"] for b in batches] + [global_acc]
    all_n      = [batch_loo[b]["n_labeled"] for b in batches] + [sum(batch_loo[b]["n_labeled"] for b in batches)]
    x = np.arange(len(all_labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 1.5), 5))
    bars1 = ax.bar(x - width/2, all_f1s,  width, label="Macro-F1",  color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width/2, all_accs, width, label="Accuracy",  color="orange",    alpha=0.85)
    for bar, n in zip(bars1, all_n):
        ax.text(bar.get_x() + bar.get_width()/2, -0.04, f"n={n}",
                ha="center", va="top", fontsize=7, color="dimgrey", rotation=45)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("LOO evaluation — grid search best params (parallel)")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_summary.png", dpi=150); plt.close()
    print("  Saved accuracy_summary.png")


def run_full(pairs, norm, k_graph, alpha, conf_threshold):
    print("\n" + "="*60)
    print("Full run with best params")
    print(f"  norm={norm}, k_graph={k_graph}, alpha={alpha}, conf={conf_threshold}")
    print("="*60)

    all_results, loo_y_true_all, loo_y_pred_all = [], [], []
    ari_scores, batch_loo = {}, {}

    for aff_path, label_path, batch in pairs:
        print(f"\nBatch: {batch}")
        idr_names, A_raw = load_affinity_matrix(aff_path)
        y = load_known_labels(label_path, idr_names)
        A_norm   = normalize_affinity(A_raw, norm)
        A_sparse = build_knn_graph(A_norm, k_graph)
        k_spread = min(k_graph, 20)
        n_labeled = int((y != -1).sum())
        print(f"  IDRs: {len(idr_names)}  |  Labeled: {n_labeled}")

        f1, acc, yt, yp = loo_accuracy(A_sparse, y, alpha, k_spread)
        if f1 is not None:
            print(f"  LOO macro-F1: {f1:.3f}  acc: {acc:.3f}")
            loo_y_true_all.extend(yt.tolist())
            loo_y_pred_all.extend(yp.tolist())
            batch_loo[batch] = {"f1": f1, "acc": acc, "n_labeled": n_labeled}
            plot_confusion_matrix(yt, yp, OUTPUT_DIR / f"{batch}_confusion.png",
                                  title=f"{batch} — LOO confusion (F1={f1:.2f})")
        else:
            print("  Skipping LOO (too few labeled samples)")

        model, pred_labels, max_proba = run_label_spreading(A_sparse, y, alpha, k_spread)
        final_labels = derive_final_labels(pred_labels, max_proba, conf_threshold)
        n_assigned = max(0, int((final_labels != UNKNOWN_LABEL).sum()) - n_labeled)
        mean_conf  = float(max_proba[final_labels != UNKNOWN_LABEL].mean())
        print(f"  Newly assigned: {n_assigned}  |  Mean confidence: {mean_conf:.3f}")

        results_df = build_results_df(idr_names, y, pred_labels, final_labels,
                                      max_proba, model.label_distributions_, model.classes_)
        results_df["batch"] = batch
        all_results.append(results_df)

        plot_class_distribution(results_df, OUTPUT_DIR / f"{batch}_class_dist.png", batch_name=batch)
        plot_confidence_by_class(results_df, OUTPUT_DIR / f"{batch}_confidence.png",
                                 conf_threshold=conf_threshold,
                                 title=f"{batch} — confidence by class")
        ari = plot_hc_vs_ls(A_sparse, results_df, OUTPUT_DIR / f"{batch}_hc_vs_ls.png", batch_name=batch)
        ari_scores[batch] = ari
        if ari is not None:
            print(f"  ARI (HC vs LS): {ari:.3f}")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
    print(f"\nWrote combined results to {OUTPUT_DIR / 'all_results.csv'}")

    if loo_y_true_all:
        plot_confusion_matrix(np.array(loo_y_true_all), np.array(loo_y_pred_all),
                              OUTPUT_DIR / "global_confusion.png",
                              title="Global LOO confusion — parallel grid search best params")
        gf1  = f1_score(loo_y_true_all, loo_y_pred_all, average="macro", zero_division=0)
        gacc = (np.array(loo_y_true_all) == np.array(loo_y_pred_all)).mean()
        print(f"Global LOO macro-F1: {gf1:.3f}  acc: {gacc:.3f}")
        plot_accuracy_summary(batch_loo, float(gf1), float(gacc), OUTPUT_DIR)

    print("\nARI scores:")
    for batch, ari in ari_scores.items():
        print(f"  {batch}: {ari:.3f}" if ari is not None else f"  {batch}: skipped")
    plot_class_distribution(combined_df, OUTPUT_DIR / "global_class_dist.png",
                            batch_name="priority batches")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    pairs = find_pairs()
    if not pairs:
        raise RuntimeError(
            f"No matching pairs found for priority batches {PRIORITY_BATCHES}.\n"
            f"Check that DATA_DIR ({DATA_DIR}) contains the CSV files."
        )
    print(f"Found {len(pairs)} priority batch(es): {[b for _,_,b in pairs]}")

    results_df = run_grid_search(pairs)

    print("\nGenerating grid search figures...")
    plot_grid_heatmaps(results_df, OUTPUT_DIR)
    plot_normalization_comparison(results_df, OUTPUT_DIR)

    norm, k_graph, alpha, conf_threshold = pick_best_params(results_df)
    run_full(pairs, norm, k_graph, alpha, conf_threshold)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("\nTo generate post-run analysis figures run:")
    print(f"  python analysis_results.py --results {OUTPUT_DIR}/all_results.csv "
          f"--output results_analysis_gridsearch_parallel")


if __name__ == "__main__":
    main()
