#!/usr/bin/env python3
"""
analysis_results.py

Post-run analysis figures for any GraphModel that produces an all_results.csv.
Reads from RESULTS_CSV (default: results/all_results.csv) and writes all
figures to OUTPUT_DIR (default: results_analysis/).

Figures produced
----------------
Global (across all batches):
  1. label_change_heatmap.png      — heatmap: rows=batch, cols=class,
                                     cells = newly assigned IDR count
                                     (mirrors analysis/class_distribution_heatmap.png
                                      but for *new* assignments, not known labels)
  2. sankey_alluvial.png           — stacked bar "before vs after" showing
                                     known / newly assigned / unknown counts
                                     per class globally
  3. confidence_global.png         — violin plot of max_proba per final class
                                     (global, all batches combined)
  4. ari_summary.png               — bar chart of ARI (HC vs LS) per batch
  5. loo_dropout_summary.png       — bar chart of LOO macro-F1 and accuracy
                                     per batch (read from results CSV metadata
                                     if present, otherwise skipped)

Per-batch (one figure per batch):
  6. <batch>_change_heatmap.png    — same as (1) but for a single batch
  7. <batch>_before_after.png      — grouped bar: known vs newly assigned
                                     vs unknown per class for that batch
  8. <batch>_confidence.png        — box plot of max_proba per class

Usage
-----
    python analysis_results.py
    python analysis_results.py --results results_norm/all_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_CSV = Path("./results/all_results.csv")
DEFAULT_OUTPUT_DIR  = Path("./results_analysis")

UNKNOWN_LABEL        = 19
CONFIDENCE_THRESHOLD = 0.6

# ARI scores are not stored in all_results.csv — they are printed to stdout.
# Paste them here manually if you want the ARI bar chart, otherwise leave
# empty and the chart will be skipped.
# Format: {"batch_name": ari_value, ...}
ARI_SCORES: dict = {}

# LOO scores — same situation. Paste from terminal output if desired.
# Format: {"batch_name": {"f1": float, "acc": float}, ...}
LOO_SCORES: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"idr", "known_label", "pred_label", "final_label",
                "max_proba", "batch"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    return df


def label_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'status' column: 'known' | 'newly assigned' | 'unknown'."""
    out = df.copy()
    out["status"] = "newly assigned"
    out.loc[out["known_label"] != -1, "status"] = "known"
    out.loc[out["final_label"] == UNKNOWN_LABEL, "status"] = "unknown"
    return out


def class_change_matrix(df: pd.DataFrame, batches: list) -> pd.DataFrame:
    """
    Returns a DataFrame (rows=batch, cols=class) with the count of
    *newly assigned* IDRs per class per batch.
    """
    new_df = df[df["status"] == "newly assigned"]
    all_classes = sorted(df[df["final_label"] != UNKNOWN_LABEL]["final_label"].unique())

    rows = {}
    for batch in batches:
        sub = new_df[new_df["batch"] == batch]
        counts = sub["final_label"].value_counts()
        rows[batch] = {cls: int(counts.get(cls, 0)) for cls in all_classes}

    return pd.DataFrame(rows, index=all_classes).T   # shape: (n_batches, n_classes)


# ---------------------------------------------------------------------------
# Figure 1 — Global label-change heatmap
# ---------------------------------------------------------------------------

def plot_label_change_heatmap(df: pd.DataFrame, batches: list, out_dir: Path):
    mat = class_change_matrix(df, batches)

    fig, ax = plt.subplots(figsize=(max(10, len(mat.columns)), max(5, len(mat))))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Newly assigned IDRs"}, ax=ax)
    ax.set_xlabel("Condensate class", fontsize=12)
    ax.set_ylabel("Batch (IDR length range)", fontsize=12)
    ax.set_title("Newly assigned IDRs per condensate class per batch\n"
                 "(excludes known labels and unknown assignments)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "label_change_heatmap.png", dpi=150)
    plt.close()
    print("  Saved label_change_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2 — Global before/after stacked bar (sankey-style)
# ---------------------------------------------------------------------------

def plot_global_before_after(df: pd.DataFrame, out_dir: Path):
    """
    Two side-by-side stacked bars per class:
      Left  = known labels (ground truth input)
      Right = final assignments (known + newly assigned, unknown excluded)
    Shows visually how much each class grew after propagation.
    """
    all_classes = sorted(df[df["final_label"] != UNKNOWN_LABEL]["final_label"].unique())

    known_counts   = df[df["known_label"] != -1]["known_label"].value_counts()
    new_counts     = df[(df["status"] == "newly assigned") &
                        (df["final_label"] != UNKNOWN_LABEL)]["final_label"].value_counts()
    unknown_counts = df[df["status"] == "unknown"]["final_label"].value_counts()

    x = np.arange(len(all_classes))
    width = 0.35

    known  = [int(known_counts.get(c, 0))   for c in all_classes]
    new    = [int(new_counts.get(c, 0))     for c in all_classes]
    unk    = [int(unknown_counts.get(c, 0)) for c in all_classes]

    fig, ax = plt.subplots(figsize=(max(10, len(all_classes) * 0.8), 6))

    # Left bar: known only
    ax.bar(x - width/2, known, width, label="Known (input)", color="steelblue")

    # Right bar: known + newly assigned stacked
    ax.bar(x + width/2, known, width, label="Known (retained)", color="steelblue", alpha=0.5)
    ax.bar(x + width/2, new,   width, bottom=known,
           label="Newly assigned", color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_classes], rotation=45)
    ax.set_xlabel("Condensate class", fontsize=12)
    ax.set_ylabel("Number of IDRs", fontsize=12)
    ax.set_title("Global label assignment: before vs after propagation\n"
                 "(left = input known labels, right = final assignments)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "global_before_after.png", dpi=150)
    plt.close()
    print("  Saved global_before_after.png")


# ---------------------------------------------------------------------------
# Figure 3 — Global confidence violin plot
# ---------------------------------------------------------------------------

def plot_global_confidence(df: pd.DataFrame, out_dir: Path):
    plot_df = df[df["final_label"] != UNKNOWN_LABEL].copy()
    plot_df["class"] = plot_df["final_label"].astype(str)
    order = [str(c) for c in sorted(plot_df["final_label"].unique())]

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.9), 5))
    sns.violinplot(data=plot_df, x="class", y="max_proba", hue="class",
                   order=order, inner="quartile", palette="muted",
                   legend=False, ax=ax)
    ax.axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--",
               linewidth=1.2, label=f"Confidence threshold ({CONFIDENCE_THRESHOLD})")
    ax.set_xlabel("Condensate class", fontsize=12)
    ax.set_ylabel("Max probability", fontsize=12)
    ax.set_title("Prediction confidence distribution per class (all batches)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_global.png", dpi=150)
    plt.close()
    print("  Saved confidence_global.png")


# ---------------------------------------------------------------------------
# Figure 4 — ARI summary bar chart
# ---------------------------------------------------------------------------

def plot_ari_summary(ari_scores: dict, out_dir: Path):
    if not ari_scores:
        print("  Skipping ari_summary.png (ARI_SCORES dict is empty — "
              "paste values from terminal output into ARI_SCORES at top of script)")
        return

    batches = list(ari_scores.keys())
    values  = [ari_scores[b] for b in batches]
    colors  = ["tomato" if v < 0 else "steelblue" for v in values]

    fig, ax = plt.subplots(figsize=(max(8, len(batches)), 4))
    bars = ax.bar(batches, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.002 if v >= 0 else -0.008),
                f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9)
    ax.set_xlabel("Batch", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title("ARI: Hierarchical Clustering vs LabelSpreading per batch\n"
                 "(higher = more agreement between methods)", fontsize=12)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "ari_summary.png", dpi=150)
    plt.close()
    print("  Saved ari_summary.png")


# ---------------------------------------------------------------------------
# Figure 5 — LOO dropout summary
# ---------------------------------------------------------------------------

def plot_loo_summary(loo_scores: dict, out_dir: Path):
    if not loo_scores:
        print("  Skipping loo_dropout_summary.png (LOO_SCORES dict is empty — "
              "paste values from terminal output into LOO_SCORES at top of script)")
        return

    batches = list(loo_scores.keys())
    f1s     = [loo_scores[b]["f1"]  for b in batches]
    accs    = [loo_scores[b]["acc"] for b in batches]

    x     = np.arange(len(batches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(batches)), 4))
    ax.bar(x - width/2, f1s,  width, label="Macro-F1",  color="steelblue", alpha=0.85)
    ax.bar(x + width/2, accs, width, label="Accuracy",  color="orange",    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("LOO evaluation per batch (macro-F1 and accuracy)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "loo_dropout_summary.png", dpi=150)
    plt.close()
    print("  Saved loo_dropout_summary.png")


# ---------------------------------------------------------------------------
# Figure 6 — Per-batch label-change heatmap (single row, all classes)
# ---------------------------------------------------------------------------

def plot_batch_change_heatmap(df: pd.DataFrame, batch: str, out_dir: Path):
    sub = df[df["batch"] == batch]
    all_classes = sorted(df[df["final_label"] != UNKNOWN_LABEL]["final_label"].unique())

    new_counts = (sub[sub["status"] == "newly assigned"]["final_label"]
                  .value_counts())
    row = pd.DataFrame(
        [[int(new_counts.get(c, 0)) for c in all_classes]],
        index=[batch],
        columns=all_classes,
    )

    fig, ax = plt.subplots(figsize=(max(10, len(all_classes)), 2))
    sns.heatmap(row, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Newly assigned"}, ax=ax)
    ax.set_xlabel("Condensate class", fontsize=11)
    ax.set_title(f"{batch} — newly assigned IDRs per class", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{batch}_change_heatmap.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7 — Per-batch before/after grouped bar
# ---------------------------------------------------------------------------

def plot_batch_before_after(df: pd.DataFrame, batch: str, out_dir: Path):
    sub = df[df["batch"] == batch]
    all_classes = sorted(df[df["final_label"] != UNKNOWN_LABEL]["final_label"].unique())

    known  = sub[sub["known_label"] != -1]["known_label"].value_counts()
    new    = sub[(sub["status"] == "newly assigned") &
                 (sub["final_label"] != UNKNOWN_LABEL)]["final_label"].value_counts()

    x     = np.arange(len(all_classes))
    width = 0.35

    k_vals = [int(known.get(c, 0)) for c in all_classes]
    n_vals = [int(new.get(c, 0))   for c in all_classes]

    fig, ax = plt.subplots(figsize=(max(10, len(all_classes) * 0.8), 5))
    ax.bar(x - width/2, k_vals, width, label="Known (input)", color="steelblue")
    ax.bar(x + width/2, k_vals, width, label="Known (retained)", color="steelblue", alpha=0.5)
    ax.bar(x + width/2, n_vals, width, bottom=k_vals,
           label="Newly assigned", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in all_classes], rotation=45)
    ax.set_xlabel("Condensate class", fontsize=11)
    ax.set_ylabel("Number of IDRs", fontsize=11)
    ax.set_title(f"{batch} — label assignment before vs after propagation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{batch}_before_after.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8 — Per-batch confidence box plot
# ---------------------------------------------------------------------------

def plot_batch_confidence(df: pd.DataFrame, batch: str, out_dir: Path):
    sub = df[(df["batch"] == batch) & (df["final_label"] != UNKNOWN_LABEL)].copy()
    if sub.empty:
        return
    order = sorted(sub["final_label"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(order)), 4))
    sns.boxplot(data=sub, x="final_label", y="max_proba", hue="final_label",
                order=order, palette="muted", legend=False, ax=ax)
    ax.axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--",
               linewidth=1.2, label=f"Threshold ({CONFIDENCE_THRESHOLD})")
    ax.set_xlabel("Condensate class", fontsize=11)
    ax.set_ylabel("Max probability", fontsize=11)
    ax.set_title(f"{batch} — prediction confidence per class", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / f"{batch}_confidence.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 9 — Global assignment summary: stacked bar per batch
# ---------------------------------------------------------------------------

def plot_global_assignment_summary(df: pd.DataFrame, batches: list, out_dir: Path):
    """
    Stacked bar per batch showing total known / newly assigned / unknown counts.
    Gives a quick overview of how much the model added vs left unknown.
    """
    known_n  = []
    new_n    = []
    unk_n    = []

    for batch in batches:
        sub = df[df["batch"] == batch]
        known_n.append(int((sub["status"] == "known").sum()))
        new_n.append(int((sub["status"] == "newly assigned").sum()))
        unk_n.append(int((sub["status"] == "unknown").sum()))

    x = np.arange(len(batches))
    fig, ax = plt.subplots(figsize=(max(10, len(batches) * 1.2), 5))
    ax.bar(x, known_n, label="Known (input)",     color="steelblue")
    ax.bar(x, new_n,   bottom=known_n,            label="Newly assigned", color="orange")
    ax.bar(x, unk_n,   bottom=[k+n for k,n in zip(known_n, new_n)],
           label="Unknown (low confidence)", color="lightgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    ax.set_xlabel("Batch (IDR length range)", fontsize=12)
    ax.set_ylabel("Number of IDRs", fontsize=12)
    ax.set_title("Label assignment summary per batch\n"
                 "(known = input labels, newly assigned = model output, "
                 "unknown = below confidence threshold)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "global_assignment_summary.png", dpi=150)
    plt.close()
    print("  Saved global_assignment_summary.png")


# ---------------------------------------------------------------------------
# Figure 10 — Global assignment summary: 100% stacked bar (after state)
# ---------------------------------------------------------------------------

def plot_global_assignment_summary_pct(df: pd.DataFrame, batches: list, out_dir: Path):
    """
    One 100% stacked bar per batch showing the after-propagation state as
    percentages of total IDRs in that batch.

    Segments:
      - Known (input)       : had a label before propagation
      - Newly assigned      : unlabeled before, got a confident label after
      - Unassigned          : still unlabeled after (class 19 OR still -1)
    """
    known_pct = []
    new_pct   = []
    unk_pct   = []

    for batch in batches:
        sub   = df[df["batch"] == batch]
        total = len(sub)
        if total == 0:
            known_pct.append(0); new_pct.append(0); unk_pct.append(100)
            continue
        known = (sub["status"] == "known").sum()
        new   = (sub["status"] == "newly assigned").sum()
        # unassigned = low-confidence (class 19) + still unlabeled (-1)
        unk   = total - known - new
        known_pct.append(known / total * 100)
        new_pct.append(new   / total * 100)
        unk_pct.append(unk   / total * 100)

    x   = np.arange(len(batches))
    fig, ax = plt.subplots(figsize=(max(10, len(batches) * 1.2), 5))

    b1 = ax.bar(x, known_pct, label="Known (input)",        color="steelblue")
    b2 = ax.bar(x, new_pct,   bottom=known_pct,             label="Newly assigned", color="orange")
    b3 = ax.bar(x, unk_pct,   bottom=[k+n for k,n in zip(known_pct, new_pct)],
                label="Unassigned (class 19 or unlabeled)", color="lightgrey")

    # Annotate each segment with its percentage if large enough to read
    for i, (k, n, u) in enumerate(zip(known_pct, new_pct, unk_pct)):
        if k > 2:
            ax.text(x[i], k / 2,         f"{k:.1f}%", ha="center", va="center", fontsize=7, color="white")
        if n > 2:
            ax.text(x[i], k + n / 2,     f"{n:.1f}%", ha="center", va="center", fontsize=7, color="black")
        if u > 2:
            ax.text(x[i], k + n + u / 2, f"{u:.1f}%", ha="center", va="center", fontsize=7, color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    ax.set_xlabel("Batch (IDR length range)", fontsize=12)
    ax.set_ylabel("Percentage of IDRs in batch (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Label assignment after propagation — % of total IDRs per batch\n"
                 "(unassigned = below confidence threshold OR never labeled)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_assignment_summary_pct.png", dpi=150)
    plt.close()
    print("  Saved global_assignment_summary_pct.png")


# ---------------------------------------------------------------------------
# Figure 11 — Global before/after: 100% stacked bars side by side per batch
# ---------------------------------------------------------------------------

def plot_global_before_after_pct(df: pd.DataFrame, batches: list, out_dir: Path):
    """
    Two 100% stacked bars per batch (before and after propagation), each
    normalised to the total IDR count in that batch.

    Before segments:
      - Known (input)   : had a label before propagation
      - Unassigned      : no label before propagation (all -1 IDRs)

    After segments:
      - Known (input)   : same known labels retained
      - Newly assigned  : gained a confident label after propagation
      - Unassigned      : still no confident label (class 19 or -1)
    """
    width = 0.35
    x     = np.arange(len(batches))

    before_known = []
    before_unk   = []
    after_known  = []
    after_new    = []
    after_unk    = []

    for batch in batches:
        sub   = df[df["batch"] == batch]
        total = len(sub)
        if total == 0:
            before_known.append(0); before_unk.append(100)
            after_known.append(0);  after_new.append(0); after_unk.append(100)
            continue

        # Before: known labels vs everything else (all unlabeled = -1)
        n_known = (sub["known_label"] != -1).sum()
        before_known.append(n_known / total * 100)
        before_unk.append((total - n_known) / total * 100)

        # After: known / newly assigned / unassigned
        n_new = (sub["status"] == "newly assigned").sum()
        n_unk = total - n_known - n_new
        after_known.append(n_known / total * 100)
        after_new.append(n_new    / total * 100)
        after_unk.append(n_unk    / total * 100)

    fig, ax = plt.subplots(figsize=(max(12, len(batches) * 1.8), 6))

    # Before bars
    ax.bar(x - width/2, before_known, width, label="Known (input)",
           color="steelblue")
    ax.bar(x - width/2, before_unk,   width, bottom=before_known,
           label="Unassigned (before)", color="lightgrey")

    # After bars
    ax.bar(x + width/2, after_known, width, label="_nolegend_",
           color="steelblue", alpha=0.6)
    ax.bar(x + width/2, after_new,   width, bottom=after_known,
           label="Newly assigned", color="orange")
    after_bottom = [k + n for k, n in zip(after_known, after_new)]
    ax.bar(x + width/2, after_unk,   width, bottom=after_bottom,
           label="Unassigned (after)", color="silver")

    # Annotate the newly assigned segment if large enough
    for i, (k, n) in enumerate(zip(after_known, after_new)):
        if n > 1.5:
            ax.text(x[i] + width/2, k + n/2, f"{n:.1f}%",
                    ha="center", va="center", fontsize=7, color="black")

    # x-axis labels with "before / after" sub-labels
    ax.set_xticks(x)
    ax.set_xticklabels(batches, rotation=30, ha="right")
    ax.set_xlabel("Batch (IDR length range)  —  left bar = before, right bar = after",
                  fontsize=11)
    ax.set_ylabel("Percentage of IDRs in batch (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Label assignment before vs after propagation — % of total IDRs per batch\n"
                 "(grey shrinks as propagation assigns more IDRs)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "global_before_after_pct.png", dpi=150)
    plt.close()
    print("  Saved global_before_after_pct.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV,
                        help="Path to all_results.csv from a GraphModel run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for figures")
    args = parser.parse_args()

    results_csv = args.results
    out_dir     = args.output

    if not results_csv.is_file():
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv}\n"
            f"Run GraphModel_03.py (or equivalent) first to generate it."
        )

    out_dir.mkdir(exist_ok=True)
    print(f"Reading results from: {results_csv}")
    print(f"Writing figures to:   {out_dir}\n")

    df = load_results(results_csv)
    df = label_status(df)

    # Sort batches in ascending IDR length order
    batch_order = ["50_99", "100_150", "151_200", "201_250",
                   "251_300", "301_350", "351_400", "401_450"]
    present = set(df["batch"].unique())
    batches = [b for b in batch_order if b in present] + \
              sorted(present - set(batch_order))
    print(f"Batches found: {batches}\n")

    # --- Global figures ---
    print("Generating global figures...")
    plot_label_change_heatmap(df, batches, out_dir)
    plot_global_before_after(df, out_dir)
    plot_global_confidence(df, out_dir)
    plot_global_assignment_summary(df, batches, out_dir)
    plot_global_assignment_summary_pct(df, batches, out_dir)
    plot_global_before_after_pct(df, batches, out_dir)
    plot_ari_summary(ARI_SCORES, out_dir)
    plot_loo_summary(LOO_SCORES, out_dir)

    # --- Per-batch figures ---
    print("\nGenerating per-batch figures...")
    for batch in batches:
        print(f"  {batch}...")
        plot_batch_change_heatmap(df, batch, out_dir)
        plot_batch_before_after(df, batch, out_dir)
        plot_batch_confidence(df, batch, out_dir)

    print(f"\nDone. All figures saved to {out_dir}/")
    print("\nNote: to include ARI and LOO bar charts, paste the values from")
    print("your terminal output into the ARI_SCORES and LOO_SCORES dicts")
    print("at the top of this script and re-run.")


if __name__ == "__main__":
    main()
