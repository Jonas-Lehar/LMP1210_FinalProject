#!/usr/bin/env python3
"""
export_idr_classes.py

Parses all_results.csv and writes one .txt file per condensate class (1-18),
plus one file for unassigned IDRs (class 19 or still unlabeled / -1).

Each txt file contains one IDR name per line, drawn from all batches.

Output folder: idr_class_lists/
  class_01.txt  ... class_18.txt   — IDRs assigned to each condensate class
  class_unassigned.txt             — IDRs with final_label == 19 or -1

Usage:
    python export_idr_classes.py
    python export_idr_classes.py --results results_final/all_results.csv
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

DEFAULT_RESULTS_CSV = Path("./results_final/all_results.csv")
DEFAULT_OUTPUT_DIR  = Path("./idr_class_lists")

UNKNOWN_LABEL = 19
VALID_CLASSES = list(range(1, 19))   # 1 through 18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_CSV,
                        help="Path to all_results.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output folder for txt files")
    args = parser.parse_args()

    if not args.results.is_file():
        raise FileNotFoundError(f"Results CSV not found: {args.results}")

    args.output.mkdir(exist_ok=True)

    df = pd.read_csv(args.results)
    if "idr" not in df.columns or "final_label" not in df.columns:
        raise ValueError("CSV must have 'idr' and 'final_label' columns.")

    # Bucket IDRs by final_label
    class_to_idrs = defaultdict(list)
    for _, row in df.iterrows():
        class_to_idrs[int(row["final_label"])].append(str(row["idr"]))

    # Write one file per valid class (1-18)
    for cls in VALID_CLASSES:
        idrs = class_to_idrs.get(cls, [])
        out_path = args.output / f"class_{cls:02d}.txt"
        out_path.write_text("\n".join(idrs) + ("\n" if idrs else ""),
                            encoding="utf-8")
        print(f"  class_{cls:02d}.txt  —  {len(idrs):5d} IDRs")

    # Write unassigned file: class 19 + any remaining -1
    unassigned = class_to_idrs.get(UNKNOWN_LABEL, []) + class_to_idrs.get(-1, [])
    out_path = args.output / "class_unassigned.txt"
    out_path.write_text("\n".join(unassigned) + ("\n" if unassigned else ""),
                        encoding="utf-8")
    print(f"  class_unassigned.txt  —  {len(unassigned):5d} IDRs "
          f"(class 19: {len(class_to_idrs.get(UNKNOWN_LABEL, []))}, "
          f"-1: {len(class_to_idrs.get(-1, []))})")

    total = len(df)
    assigned = sum(len(class_to_idrs[c]) for c in VALID_CLASSES)
    print(f"\nTotal IDRs: {total}  |  Assigned: {assigned}  |  "
          f"Unassigned: {len(unassigned)}")
    print(f"Output saved to {args.output}/")


if __name__ == "__main__":
    main()
