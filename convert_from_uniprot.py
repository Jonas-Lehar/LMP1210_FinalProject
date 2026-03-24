import pandas as pd
import ast
import re

# ---------------------------
# Input files
# ---------------------------
idr_file = "data/201_250.csv"              # can be txt/csv/tsv
condensate_file = "MSA_results_updated_colnum.csv"  # or change to .xlsx below
output_file = "Cleaned_Data/201_250_test.csv"

# ---------------------------
# Read IDR input
# Keep only the first column, drop the rest
# ---------------------------
idr_df = pd.read_csv(idr_file, sep=None, engine="python")
idr_df = idr_df.iloc[:, [0]].copy()
idr_df.columns = ["idr"]

# Clean whitespace
idr_df["idr"] = idr_df["idr"].astype(str).str.strip()

# Drop empty rows and possible repeated header row
idr_df = idr_df[idr_df["idr"].notna()]
idr_df = idr_df[idr_df["idr"] != ""]
idr_df = idr_df[idr_df["idr"].str.lower() != "idr"]

# ---------------------------
# Read condensate table
# ---------------------------
# For Excel instead, use:
# cond_df = pd.read_excel("condensates.xlsx")
cond_df = pd.read_csv(condensate_file)
cond_df.columns = cond_df.columns.str.strip()

# ---------------------------
# Parse "Verified Plant Match Uniprot ID"
# ---------------------------
def extract_uniprot_ids(cell):
    if pd.isna(cell):
        return []

    text = str(cell).strip()
    if not text:
        return []

    # Pull UniProt-like accessions from the full flattened string
    ids = re.findall(r"\b[A-Z0-9]{6,10}\b", text)

    # Optional: deduplicate while preserving order
    seen = set()
    out = []
    for uid in ids:
        if uid not in seen:
            seen.add(uid)
            out.append(uid)

    return out

# ---------------------------
# Build lookup: UniProt ID -> Condensate #
# ---------------------------
uniprot_to_cond_num = {}

for _, row in cond_df.iterrows():
    cond_num = row.get("Condensate #", None)
    match_ids_cell = row.get("Verified Plant Match Uniprot ID", None)

    ids = extract_uniprot_ids(match_ids_cell)

    for uid in ids:
        if uid not in uniprot_to_cond_num and pd.notna(cond_num):
            try:
                uniprot_to_cond_num[uid] = int(cond_num)
            except Exception:
                pass

# Debug: print condensate numbers found in lookup
print("Condensate numbers found in condensate file:", sorted(set(uniprot_to_cond_num.values())))

# Debug: print the UniProt IDs that were loaded
print("\nUniProt IDs found in condensate file:")
for uid, cond_num in sorted(uniprot_to_cond_num.items()):
    print(f"{uid} -> {cond_num}")

# ---------------------------
# Assign condensate number
# Match if UniProt ID appears anywhere inside full IDR string
# Use -1 if no match
# ---------------------------
def find_condensate_from_full_idr(idr_name, mapping, debug=False):
    matches = [(uid, cond_num) for uid, cond_num in mapping.items() if uid in idr_name]

    if debug:
        print(f"\nIDR: {idr_name}")
        print(f"Matches found: {matches}")

    if len(matches) == 0:
        return -1
    elif len(matches) == 1:
        return matches[0][1]
    else:
        print(f"WARNING: multiple matches for {idr_name}: {matches}")
        return matches[0][1]

idr_df["condensate"] = idr_df["idr"].apply(
    lambda x: find_condensate_from_full_idr(str(x), uniprot_to_cond_num, debug=False)
).astype(int)

# Keep only final output columns
idr_df = idr_df[["idr", "condensate"]]

# Save
idr_df.to_csv(output_file, index=False)

print("\nFinal output:")
print(idr_df)
print(f"\nSaved to: {output_file}")
