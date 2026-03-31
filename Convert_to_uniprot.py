#!/usr/bin/env python3
"""
Convert_to_uniprot.py
---------------------
Takes a TXT file of comma-separated gene names (Arabidopsis thaliana),
queries the UniProt REST API to resolve each gene name to its UniProt
accession(s), and appends a new row to MSA_chat_small.csv (or a target CSV)
in the format expected by convert_from_uniprot.py.

Input TXT format (one line, comma-separated):
    FCA, FLL2, ELF3, COP1

Output: a new row appended to the target MSA CSV with:
    - Condensate #   : user-specified integer
    - CONCATTED      : list-of-lists string matching the format in MSA_chat_small.csv

Usage
-----
Edit the INPUT / OUTPUT configuration section below, then run:
    python Convert_to_uniprot.py

Requirements
------------
    pip install requests pandas
"""

import re
import time
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Path to the TXT file containing comma-separated gene names
INPUT_TXT = "gene_names.txt"

# Condensate number to assign to this set of genes
CONDENSATE_NUMBER = 19   # change to the correct condensate #

# Target MSA CSV to append the new row to
# Must already exist and have the same columns as MSA_chat_small.csv
TARGET_CSV = "MSA_chat_small.csv"

# Optional: name for the condensate (used in 'Biomolecular Condensates' column)
CONDENSATE_NAME = ""

# Organism: Arabidopsis thaliana taxonomy ID
TAXON_ID = "3702"

# Delay between UniProt API calls (seconds) — be polite to the API
API_DELAY = 0.3

# ---------------------------------------------------------------------------
# UniProt REST API query
# ---------------------------------------------------------------------------

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def query_uniprot(gene_name: str, taxon_id: str = TAXON_ID) -> list[dict]:
    """
    Query UniProt for all entries matching gene_name in organism taxon_id.

    Returns a list of dicts with keys:
        accession  : str  (e.g. 'Q9LQX2')
        entry_name : str  (e.g. 'MPSR1_ARATH')
        protein    : str  (full protein name)
        gene       : str  (primary gene name from UniProt)
        reviewed   : bool (True = Swiss-Prot, False = TrEMBL)
    """
    params = {
        "query": f"gene:{gene_name} AND taxonomy_id:{taxon_id}",
        "format": "json",
        "fields": "accession,id,protein_name,gene_names,reviewed",
        "size": 50,
    }

    try:
        resp = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: UniProt query failed for '{gene_name}': {e}")
        return []

    results = []
    for entry in resp.json().get("results", []):
        accession  = entry.get("primaryAccession", "")
        entry_name = entry.get("uniProtkbId", "")
        reviewed   = entry.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

        # Protein name
        pn = entry.get("proteinDescription", {})
        rec = pn.get("recommendedName", {}) or pn.get("submittedName", [{}])
        if isinstance(rec, list):
            rec = rec[0] if rec else {}
        protein = rec.get("fullName", {}).get("value", "")

        # Primary gene name
        genes = entry.get("genes", [])
        gene  = genes[0].get("geneName", {}).get("value", "") if genes else ""

        results.append({
            "accession":  accession,
            "entry_name": entry_name,
            "protein":    protein,
            "gene":       gene,
            "reviewed":   reviewed,
        })

    return results


def build_full_entry_string(hit: dict) -> str:
    """
    Build the full UniProt entry string as it appears in MSA_chat_small.csv.
    Format: ' sp|ACCESSION|ENTRY_NAME PROTEIN_NAME OS=Arabidopsis thaliana OX=3702 GN=GENE PE=1 SV=1'
    """
    prefix = "sp" if hit["reviewed"] else "tr"
    return (
        f" {prefix}|{hit['accession']}|{hit['entry_name']}"
        f" {hit['protein']}"
        f" OS=Arabidopsis thaliana OX=3702"
        f" GN={hit['gene']}"
        f" PE=1 SV=1"
    )


# ---------------------------------------------------------------------------
# Build CONCATTED string
# ---------------------------------------------------------------------------

def build_concatted(gene_names: list[str]) -> tuple[str, str]:
    """
    For each gene name, query UniProt and build:
      - verified_match : human-readable string  (for 'Verified Plant Match' column)
      - concatted      : list-of-lists string   (for 'CONCATTED' column)

    Returns (verified_match_str, concatted_str)
    """
    parts_full   = []   # for 'Verified Plant Match' column
    parts_concat = []   # for 'CONCATTED' column

    for gene in gene_names:
        gene = gene.strip()
        if not gene:
            continue

        print(f"  Querying UniProt for: {gene}")
        hits = query_uniprot(gene)
        time.sleep(API_DELAY)

        full_strings = [build_full_entry_string(h) for h in hits]
        accessions   = [h["accession"] for h in hits]

        if hits:
            print(f"    -> {len(hits)} result(s): {accessions[:3]}{'...' if len(accessions) > 3 else ''}")
        else:
            print(f"    -> No results found")

        parts_full.append(f"'{gene}', {full_strings}")
        parts_concat.append(f"'{gene}', {accessions}")

    # Format to match the existing CONCATTED column style:
    # ['GENE1', ['ACC1', 'ACC2'], 'GENE2', ['ACC3'], ...]
    concat_items = []
    full_items   = []
    for gene in gene_names:
        gene = gene.strip()
        if not gene:
            continue
        hits       = query_uniprot(gene)
        time.sleep(API_DELAY)
        accessions = [h["accession"] for h in hits]
        full_strs  = [build_full_entry_string(h) for h in hits]
        concat_items.extend([f"'{gene}'", str(accessions)])
        full_items.extend([f"'{gene}'", str(full_strs)])

    concatted_str = "[" + ", ".join(concat_items) + "]"
    full_str      = "[" + ", ".join(full_items) + "]"
    return full_str, concatted_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Read gene names ---
    txt_path = Path(INPUT_TXT)
    if not txt_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {txt_path}\n"
            f"Create a file named '{INPUT_TXT}' with comma-separated gene names."
        )

    raw = txt_path.read_text(encoding="utf-8")
    gene_names = [g.strip() for g in raw.replace("\n", ",").split(",") if g.strip()]

    print(f"Gene names to convert ({len(gene_names)}): {gene_names}")
    print(f"Querying UniProt (taxon {TAXON_ID})...\n")

    # --- Query UniProt for each gene (two passes: full strings + accessions) ---
    # Build both columns in one pass to avoid double API calls
    concat_items = []
    full_items   = []
    all_accessions = []   # flat list for quick reference

    for gene in gene_names:
        print(f"  Querying: {gene}")
        hits = query_uniprot(gene)
        time.sleep(API_DELAY)

        accessions = [h["accession"] for h in hits]
        full_strs  = [build_full_entry_string(h) for h in hits]

        if hits:
            print(f"    -> {len(hits)} result(s): {accessions[:3]}{'...' if len(accessions) > 3 else ''}")
        else:
            print(f"    -> No results found")

        concat_items.extend([f"'{gene}'", str(accessions)])
        full_items.extend([f"'{gene}'", str(full_strs)])
        all_accessions.extend(accessions)

    concatted_str        = "[" + ", ".join(concat_items) + "]"
    verified_match_str   = "[" + ", ".join(full_items) + "]"
    verified_plant_str   = ", ".join(gene_names)

    print(f"\nResolved {len(all_accessions)} UniProt accession(s) total.")
    print(f"Accessions: {all_accessions[:10]}{'...' if len(all_accessions) > 10 else ''}")

    # --- Load target CSV and append new row ---
    csv_path = Path(TARGET_CSV)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Target CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    new_row = {col: "" for col in df.columns}
    new_row["Condensate #"]                  = CONDENSATE_NUMBER
    new_row["Biomolecular Condensates"]      = CONDENSATE_NAME
    new_row["Verified Plant"]                = verified_plant_str
    new_row["Verified Plant Match"]          = verified_match_str
    new_row["Verified Plant Match Uniprot ID"] = concatted_str
    # 'CONCATTED' is the column used by convert_from_uniprot.py
    if "CONCATTED" in df.columns:
        new_row["CONCATTED"] = concatted_str

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

    print(f"\nAppended new row (Condensate #{CONDENSATE_NUMBER}) to {csv_path}")
    print("Done. You can now run convert_from_uniprot.py.")


if __name__ == "__main__":
    main()
