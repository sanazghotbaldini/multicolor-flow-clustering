"""
generate_dataset.py
===================================================================
Create a small, realistic, *simulated* high-dimensional cytometry
dataset so this whole repository runs end-to-end on any machine with
no data download required.

The design intentionally mirrors a classic real immunology benchmark
(the Bodenmiller "BCR-XL" PBMC experiment): peripheral blood immune
cells from several patients, each measured in two conditions --
"Reference" (unstimulated) and "BCRXL" (stimulated through the B-cell
receptor). The biologically meaningful signal we plant is that the
signalling marker pS6 goes UP specifically in B cells after
stimulation -- exactly the kind of pharmacodynamic readout a
translational-medicine team looks for.

NOTE: This is synthetic data, generated for a fully reproducible
demo. The *identical* analysis pipeline (src/run_analysis.py) runs on
real .fcs files -- see the README for how to point it at the public
Bodenmiller dataset via the R package HDCytoData / FlowRepository.

Output written to ./data/:
  - sample_<id>.csv  : one file per sample (rows = cells, cols = markers)
  - metadata.csv     : sample_id, patient_id, condition
  - panel.csv        : marker, marker_class ("type" or "state")
===================================================================
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Reproducibility: fix the random seed so everyone gets identical data
# ------------------------------------------------------------------
RNG = np.random.default_rng(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Define the antibody panel
#    "type"  markers  -> used to DEFINE cell populations (clustering)
#    "state" markers  -> functional/signalling, compared between groups
# ------------------------------------------------------------------
PANEL = [
    ("CD3",    "type"),   # T cells
    ("CD4",    "type"),   # helper T cells
    ("CD8",    "type"),   # cytotoxic T cells
    ("CD19",   "type"),   # B cells
    ("CD20",   "type"),   # B cells
    ("CD56",   "type"),   # NK cells
    ("CD16",   "type"),   # NK / monocytes
    ("CD14",   "type"),   # monocytes
    ("CD11c",  "type"),   # myeloid / dendritic cells
    ("CD123",  "type"),   # plasmacytoid DC / basophils
    ("HLA_DR", "type"),   # antigen-presenting cells
    ("CD45RA", "type"),   # naive vs memory
    ("pS6",    "state"),  # SIGNALLING readout (the planted effect)
    ("pNFkB",  "state"),  # second signalling readout
]
MARKERS = [m for m, _ in PANEL]
TYPE_MARKERS = [m for m, c in PANEL if c == "type"]

# ------------------------------------------------------------------
# 2. Define ground-truth cell populations.
#    For each population we give a mean expression for every marker.
#    Values are on an "intensity" scale (roughly 0-8 after the lab's
#    usual arcsinh view). HIGH ~= positive, LOW ~= background.
#    These are the labels we will pretend NOT to know, then try to
#    recover with unsupervised clustering.
# ------------------------------------------------------------------
HI, MID, LO = 5.0, 3.0, 0.5   # high / intermediate / background levels

def profile(**kw):
    """Build a full marker mean-vector, defaulting every marker to LO."""
    base = {m: LO for m in MARKERS}
    base.update(kw)
    return base

POPULATIONS = {
    # name            frequency   marker means
    "CD4_T":  dict(freq=0.32, mean=profile(CD3=HI, CD4=HI,  CD45RA=MID)),
    "CD8_T":  dict(freq=0.22, mean=profile(CD3=HI, CD8=HI,  CD45RA=MID)),
    "B_cell": dict(freq=0.14, mean=profile(CD19=HI, CD20=HI, HLA_DR=HI)),
    "NK":     dict(freq=0.12, mean=profile(CD56=HI, CD16=HI)),
    "Mono":   dict(freq=0.16, mean=profile(CD14=HI, CD11c=MID, HLA_DR=MID)),
    "DC":     dict(freq=0.04, mean=profile(CD11c=HI, HLA_DR=HI, CD123=MID)),
}

# ------------------------------------------------------------------
# 3. Experimental design: 8 patients x 2 conditions = 16 samples
# ------------------------------------------------------------------
N_PATIENTS = 8
CONDITIONS = ["Reference", "BCRXL"]
CELLS_PER_SAMPLE = 5000

def sample_population(name, n, condition, induction):
    """Draw n cells from one population as a (n x n_markers) matrix.

    `induction` is a per-sample pS6 level for stimulated B cells, so the
    strength of the response varies from patient to patient (as in real
    experiments) instead of being identical everywhere.
    """
    spec = POPULATIONS[name]
    means = np.array([spec["mean"][m] for m in MARKERS], dtype=float)

    # ---- plant the biology -------------------------------------
    # pS6 rises in B cells specifically under BCRXL stimulation, with a
    # patient-specific magnitude.
    if condition == "BCRXL" and name == "B_cell":
        means[MARKERS.index("pS6")] = induction
        means[MARKERS.index("pNFkB")] = induction * 0.6
    # ------------------------------------------------------------

    # Gaussian noise around the mean; gives realistic within-population
    # spread so clusters overlap a little (like real data).
    spread = 0.7
    cells = RNG.normal(loc=means, scale=spread, size=(n, len(MARKERS)))
    cells = np.clip(cells, 0, None)            # intensities are non-negative
    return cells

def build_sample(patient, condition):
    """Assemble one full sample by mixing all populations."""
    rows, labels = [], []
    # small per-sample wobble in population frequencies (biological + technical)
    freqs = np.array([POPULATIONS[p]["freq"] for p in POPULATIONS])
    freqs = freqs * RNG.normal(1.0, 0.08, size=len(freqs))
    freqs = freqs / freqs.sum()
    counts = RNG.multinomial(CELLS_PER_SAMPLE, freqs)

    # patient-specific strength of the B-cell pS6 response
    induction = RNG.normal(HI, 0.5)

    for name, n in zip(POPULATIONS, counts):
        rows.append(sample_population(name, n, condition, induction))
        labels += [name] * n

    X = np.vstack(rows)
    # mild per-sample "batch effect": a small global intensity shift
    X = X + RNG.normal(0.0, 0.15)
    X = np.clip(X, 0, None)
    df = pd.DataFrame(X, columns=MARKERS)
    df["true_label"] = labels          # ground truth (for our own checking only)
    return df.sample(frac=1, random_state=patient).reset_index(drop=True)

# ------------------------------------------------------------------
# 4. Generate and write everything to disk
# ------------------------------------------------------------------
meta_rows = []
sample_id = 0
for patient in range(1, N_PATIENTS + 1):
    for condition in CONDITIONS:
        sample_id += 1
        sid = f"S{sample_id:02d}"
        df = build_sample(patient, condition)
        df.to_csv(os.path.join(DATA_DIR, f"sample_{sid}.csv"), index=False)
        meta_rows.append(dict(sample_id=sid, patient_id=f"P{patient}",
                              condition=condition))

pd.DataFrame(meta_rows).to_csv(os.path.join(DATA_DIR, "metadata.csv"), index=False)
pd.DataFrame(PANEL, columns=["marker", "marker_class"]).to_csv(
    os.path.join(DATA_DIR, "panel.csv"), index=False)

n_files = len([f for f in os.listdir(DATA_DIR) if f.startswith("sample_")])
print(f"Done. Wrote {n_files} sample files + metadata.csv + panel.csv to {DATA_DIR}")
print(f"Total cells: {n_files * CELLS_PER_SAMPLE:,}")
