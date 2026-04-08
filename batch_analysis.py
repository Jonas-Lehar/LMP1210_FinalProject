#!/usr/bin/env python3
"""
batch_analysis.py
-----------------
Analyses all affinity matrix batches and their label files.
Produces:
  - analysis/batch_stats.txt   : per-batch and global statistics
  - analysis/*.png             : figures for each stat category

Usage: python batch_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR    = Path("./data")
LABELS_DIR  = Path("./Cleaned_Data")
OUTPUT_DIR  = Path("./analysis")
STATS_FILE  = OUTPUT_DIR / "batch_stats.txt"

UNKNOWN_CONDENSATE = -1   # value used for unlabeled IDRs in label files
N_CONDENSATES      = 18   # total possible condensate classes (1-18)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_pairs():
    pairs = []
    for aff_path in sorted(DATA_DIR.glob("*.csv")):
        if "test" in aff_path.name:
            continue
        label_path = LABELS_DIR / f"{aff_path.stem}_test.csv"
        if label_path.is_file():
            pairs.append((aff_path, label_path, aff_path.stem))
    return pairs


def load_batch(aff_path: Path, label_path: Path):
    """Load affinity matrix (header only for IDR names) and label file."""
    # Read only the header row to get IDR names — avoids loading huge matrix
    header = pd.read_csv(aff_path, index_col=0, nrows=0)
    idr_names = header.columns.tolist()
    n_total = len(idr_names)

    labels_df = pd.read_csv(label_path)
    label_map = dict(zip(labels_df["idr"], labels_df["condensate"]))

    y = np.array([label_map.get(name, UNKNOWN_CONDENSATE) for name in idr_names])
    return idr_names, y, n_total


def load_affinity_diagonal_stats(aff_path: Path):
    """
    Load the full affinity matrix and compute summary stats on the
    off-diagonal values (pairwise affinities). Skips matrices > 3000
    to keep runtime reasonable.
    """
    header = pd.read_csv(aff_path, index_col=0, nrows=0)
    n = len(header.columns)
    if n > 10000:
        return None   # too large, skip

    df = pd.read_csv(aff_path, index_col=0)
    A  = df.to_numpy(dtype=float)
    A  = 0.5 * (A + A.T)
    np.fill_diagonal(A, np.nan)

    vals = A[~np.isnan(A)]
    return {
        "aff_mean":   float(np.mean(vals)),
        "aff_median": float(np.median(vals)),
        "aff_std":    float(np.std(vals)),
        "aff_p95":    float(np.percentile(vals, 95)),
        "aff_p05":    float(np.percentile(vals, 5)),
    }


# ---------------------------------------------------------------------------
# Per-batch statistics
# ---------------------------------------------------------------------------

def compute_batch_stats(batch, idr_names, y, n_total, aff_stats):
    labeled_mask = y != UNKNOWN_CONDENSATE
    n_labeled    = labeled_mask.sum()
    n_unlabeled  = n_total - n_labeled
    label_density = n_labeled / n_total if n_total > 0 else 0.0

    labeled_classes = y[labeled_mask]
    class_counts    = pd.Series(labeled_classes).value_counts().sort_index()
    n_classes_present = len(class_counts)

    # Shannon entropy of class distribution (higher = more balanced)
    if n_labeled > 0:
        probs = class_counts.values / class_counts.values.sum()
        class_entropy = float(entropy(probs, base=2))
        max_entropy   = float(np.log2(n_classes_present)) if n_classes_present > 1 else 0.0
        balance_ratio = class_entropy / max_entropy if max_entropy > 0 else 0.0
        dominant_class = int(class_counts.idxmax())
        dominant_frac  = float(class_counts.max() / n_labeled)
    else:
        class_entropy = balance_ratio = dominant_frac = 0.0
        dominant_class = -1

    # Imbalance ratio: largest class / smallest class
    imbalance_ratio = (float(class_counts.max() / class_counts.min())
                       if n_classes_present > 1 else float("inf"))

    stats = {
        "batch":            batch,
        "n_total":          n_total,
        "n_labeled":        int(n_labeled),
        "n_unlabeled":      int(n_unlabeled),
        "label_density":    round(label_density * 100, 2),   # as %
        "n_classes":        n_classes_present,
        "class_entropy":    round(class_entropy, 3),
        "balance_ratio":    round(balance_ratio, 3),          # 0=worst, 1=perfect
        "dominant_class":   dominant_class,
        "dominant_frac":    round(dominant_frac * 100, 2),    # as %
        "imbalance_ratio":  round(imbalance_ratio, 1),
        "class_counts":     class_counts.to_dict(),
    }
    if aff_stats:
        stats.update(aff_stats)
    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_label_density(all_stats, out_dir):
    batches = [s["batch"] for s in all_stats]
    density = [s["label_density"] for s in all_stats]
    n_total = [s["n_total"] for s in all_stats]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(batches, density, color="steelblue", alpha=0.85, label="Label density (%)")
    ax1.set_ylabel("Labeled IDRs (%)", fontsize=12)
    ax1.set_xlabel("Batch", fontsize=12)
    ax1.set_title("Label density per batch", fontsize=13)
    ax1.set_ylim(0, max(density) * 1.3)
    for bar, d in zip(bars, density):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{d:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(batches, n_total, color="darkorange", marker="o",
             linewidth=2, markersize=6, label="Total IDRs")
    ax2.set_ylabel("Total IDRs (log scale)", fontsize=12)
    ax2.set_yscale("log")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "label_density.png", dpi=150)
    plt.close()


def plot_class_distribution_heatmap(all_stats, out_dir):
    """Heatmap: rows = batches, cols = condensate classes, values = count."""
    all_classes = sorted({cls for s in all_stats for cls in s["class_counts"]})
    batches     = [s["batch"] for s in all_stats]

    matrix = np.zeros((len(batches), len(all_classes)))
    for i, s in enumerate(all_stats):
        for j, cls in enumerate(all_classes):
            matrix[i, j] = s["class_counts"].get(cls, 0)

    df = pd.DataFrame(matrix, index=batches, columns=[f"C{c}" for c in all_classes])

    fig, ax = plt.subplots(figsize=(max(10, len(all_classes)), max(5, len(batches))))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Labeled IDR count"}, ax=ax)
    ax.set_xlabel("Condensate class", fontsize=12)
    ax.set_ylabel("Batch", fontsize=12)
    ax.set_title("Labeled IDR count per condensate class per batch", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution_heatmap.png", dpi=150)
    plt.close()


def plot_class_balance(all_stats, out_dir):
    batches       = [s["batch"] for s in all_stats]
    balance       = [s["balance_ratio"] for s in all_stats]
    dominant_frac = [s["dominant_frac"] for s in all_stats]
    n_classes     = [s["n_classes"] for s in all_stats]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Balance ratio
    axes[0].bar(batches, balance, color="mediumseagreen", alpha=0.85)
    axes[0].set_ylabel("Balance ratio (0=worst, 1=perfect)", fontsize=10)
    axes[0].set_title("Class balance ratio", fontsize=11)
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    axes[0].set_xticks(range(len(batches)))
    axes[0].set_xticklabels(batches, rotation=30, ha="right")

    # Dominant class fraction
    axes[1].bar(batches, dominant_frac, color="tomato", alpha=0.85)
    axes[1].set_ylabel("Dominant class fraction (%)", fontsize=10)
    axes[1].set_title("Fraction of labels in dominant class", fontsize=11)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(100/N_CONDENSATES, color="grey", linestyle="--",
                    linewidth=0.8, label="Uniform baseline")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks(range(len(batches)))
    axes[1].set_xticklabels(batches, rotation=30, ha="right")

    # Number of classes present
    axes[2].bar(batches, n_classes, color="mediumpurple", alpha=0.85)
    axes[2].set_ylabel("Number of condensate classes present", fontsize=10)
    axes[2].set_title("Class coverage per batch", fontsize=11)
    axes[2].axhline(N_CONDENSATES, color="grey", linestyle="--",
                    linewidth=0.8, label=f"Max ({N_CONDENSATES})")
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(range(len(batches)))
    axes[2].set_xticklabels(batches, rotation=30, ha="right")

    plt.suptitle("Class imbalance metrics per batch", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_affinity_stats(all_stats, out_dir):
    """Plot mean/std of pairwise affinity scores for batches where available."""
    valid = [s for s in all_stats if "aff_mean" in s]
    if not valid:
        return

    batches = [s["batch"] for s in valid]
    means   = [s["aff_mean"]   for s in valid]
    stds    = [s["aff_std"]    for s in valid]
    p05     = [s["aff_p05"]    for s in valid]
    p95     = [s["aff_p95"]    for s in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(batches))
    ax.bar(x, means, yerr=stds, color="cornflowerblue", alpha=0.85,
           capsize=4, label="Mean ± std")
    ax.plot(x, p05, "v", color="navy",   markersize=7, label="5th percentile")
    ax.plot(x, p95, "^", color="orange", markersize=7, label="95th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    ax.set_ylabel("Pairwise affinity score", fontsize=12)
    ax.set_title("Affinity score distribution per batch\n(large batches skipped)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "affinity_stats.png", dpi=150)
    plt.close()


def plot_global_class_totals(all_stats, out_dir):
    """Bar chart of total labeled IDRs per condensate class across all batches."""
    totals = {}
    for s in all_stats:
        for cls, cnt in s["class_counts"].items():
            totals[cls] = totals.get(cls, 0) + cnt

    classes = sorted(totals.keys())
    counts  = [totals[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), 5))
    bars = ax.bar([f"C{c}" for c in classes], counts, color="steelblue", alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Condensate class", fontsize=12)
    ax.set_ylabel("Total labeled IDRs (all batches)", fontsize=12)
    ax.set_title("Global labeled IDR count per condensate class", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "global_class_totals.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_stats_report(all_stats, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("BATCH ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    total_idrs    = sum(s["n_total"]   for s in all_stats)
    total_labeled = sum(s["n_labeled"] for s in all_stats)
    global_density = total_labeled / total_idrs * 100 if total_idrs else 0

    lines.append("GLOBAL SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total batches          : {len(all_stats)}")
    lines.append(f"  Total IDRs             : {total_idrs:,}")
    lines.append(f"  Total labeled IDRs     : {total_labeled:,}")
    lines.append(f"  Global label density   : {global_density:.2f}%")

    # Global class totals
    totals = {}
    for s in all_stats:
        for cls, cnt in s["class_counts"].items():
            totals[cls] = totals.get(cls, 0) + cnt
    lines.append(f"  Classes represented    : {sorted(totals.keys())}")
    lines.append(f"  Classes missing        : "
                 f"{sorted(set(range(1, N_CONDENSATES+1)) - set(totals.keys()))}")
    lines.append("")

    lines.append("GLOBAL CLASS DISTRIBUTION")
    lines.append("-" * 40)
    for cls in sorted(totals.keys()):
        pct = totals[cls] / total_labeled * 100 if total_labeled else 0
        lines.append(f"  Class {cls:>2}: {totals[cls]:>4} IDRs  ({pct:.1f}%)")
    lines.append("")

    lines.append("PER-BATCH STATISTICS")
    lines.append("-" * 40)
    for s in all_stats:
        lines.append(f"\n  Batch: {s['batch']}")
        lines.append(f"    Total IDRs         : {s['n_total']:,}")
        lines.append(f"    Labeled IDRs       : {s['n_labeled']}  ({s['label_density']}%)")
        lines.append(f"    Unlabeled IDRs     : {s['n_unlabeled']:,}")
        lines.append(f"    Classes present    : {s['n_classes']}  {sorted(s['class_counts'].keys())}")
        lines.append(f"    Dominant class     : {s['dominant_class']}  ({s['dominant_frac']}% of labels)")
        lines.append(f"    Imbalance ratio    : {s['imbalance_ratio']}x  (max/min class count)")
        lines.append(f"    Balance ratio      : {s['balance_ratio']}  (0=worst, 1=perfect)")
        lines.append(f"    Class entropy      : {s['class_entropy']:.3f} bits")
        lines.append(f"    Class counts       : {dict(sorted(s['class_counts'].items()))}")
        if "aff_mean" in s:
            lines.append(f"    Affinity mean      : {s['aff_mean']:.4f}")
            lines.append(f"    Affinity std       : {s['aff_std']:.4f}")
            lines.append(f"    Affinity 5th pct   : {s['aff_p05']:.4f}")
            lines.append(f"    Affinity 95th pct  : {s['aff_p95']:.4f}")
        else:
            lines.append(f"    Affinity stats     : skipped (matrix too large)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("NOTES")
    lines.append("-" * 40)
    lines.append("  Balance ratio: Shannon entropy of class distribution normalised")
    lines.append("  by log2(n_classes). 1.0 = perfectly uniform, 0.0 = single class.")
    lines.append("  Imbalance ratio: count of largest class / count of smallest class.")
    lines.append("  Affinity stats computed on off-diagonal values only.")
    lines.append("  Batches with >10000 IDRs skipped for affinity stats (runtime).")
    lines.append("=" * 70)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote stats report to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    pairs = find_pairs()
    if not pairs:
        raise RuntimeError("No batch pairs found.")

    all_stats = []
    for aff_path, label_path, batch in pairs:
        print(f"Analysing {batch}...")
        idr_names, y, n_total = load_batch(aff_path, label_path)
        aff_stats = load_affinity_diagonal_stats(aff_path)
        stats = compute_batch_stats(batch, idr_names, y, n_total, aff_stats)
        all_stats.append(stats)
        print(f"  {n_total:,} IDRs, {stats['n_labeled']} labeled "
              f"({stats['label_density']}%), {stats['n_classes']} classes")

    # Sort largest to smallest so 50_99 is leftmost in all figures
    all_stats.sort(key=lambda s: s["n_total"], reverse=True)

    print("\nGenerating figures...")
    plot_label_density(all_stats, OUTPUT_DIR)
    plot_class_distribution_heatmap(all_stats, OUTPUT_DIR)
    plot_class_balance(all_stats, OUTPUT_DIR)
    plot_affinity_stats(all_stats, OUTPUT_DIR)
    plot_global_class_totals(all_stats, OUTPUT_DIR)

    write_stats_report(all_stats, STATS_FILE)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
