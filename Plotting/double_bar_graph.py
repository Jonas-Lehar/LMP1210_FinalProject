import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("./idr_labelspreading_results.csv")

# --- Label → Condensate name mapping ---
label_map = {
    -1: "Unknown",
    1: "Nucleolus",
    2: "Nuclear Body",
    6: "Nuclear Speckle",
    7: "P-body",
    16: "Stress Granule",
    19: "Cajal Body"
}

# --- Filter out unknowns from known labels ---
df_known = df[df["known_label"] != -1]

# --- Compute proportions ---
known_props = df_known["known_label"].value_counts(normalize=True)
final_props = df["final_label"].value_counts(normalize=True)

# --- Get all labels present ---
all_labels = sorted(set(known_props.index).union(final_props.index))

# --- Align proportions ---
known_props = known_props.reindex(all_labels, fill_value=0)
final_props = final_props.reindex(all_labels, fill_value=0)

# --- Convert labels to names ---
label_names = [label_map.get(l, f"Class {l}") for l in all_labels]

# --- Plot ---
x = range(len(all_labels))
width = 0.4

plt.figure(figsize=(8, 5))

plt.bar([i - width/2 for i in x], known_props, 
        width=width, label="Known")

plt.bar([i + width/2 for i in x], final_props, 
        width=width, label="Final")

# --- Formatting ---
plt.xticks(x, label_names, rotation=30, ha="right")
plt.xlabel("Condensate Type")
plt.ylabel("Proportion")
plt.title("Proportional Change in Condensate Classification")
plt.legend()
plt.tight_layout()

# --- Save + show ---
plt.savefig("condensate_proportions.png", dpi=300)
plt.show()


# --- OPTIONAL: Plot proportional change (delta) ---
change = final_props - known_props

plt.figure(figsize=(8, 5))
plt.bar(label_names, change)
plt.axhline(0)

plt.xticks(rotation=30, ha="right")
plt.xlabel("Condensate Type")
plt.ylabel("Change (Final - Known)")
plt.title("Change in Condensate Proportions")

plt.tight_layout()
plt.savefig("condensate_change.png", dpi=300)
plt.show()