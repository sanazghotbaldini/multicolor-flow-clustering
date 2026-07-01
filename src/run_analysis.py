"""
run_analysis.py
===================================================================
Unsupervised analysis of high-dimensional cytometry data.

This is the workhorse of the repository. Given a folder of per-sample
expression tables, it:

  1. Loads and pools all cells, attaching sample/condition metadata.
  2. Transforms the data the way cytometrists do (arcsinh, cofactor 5).
  3. Clusters cells WITHOUT manual gating, using the FlowSOM approach:
     a Self-Organising Map (SOM) for high-resolution clustering, then
     hierarchical "meta-clustering" to a small, interpretable number
     of populations.
  4. Annotates each meta-cluster by its marker profile (heatmap).
  5. Projects cells into 2D with UMAP for visualisation.
  6. Tests for DIFFERENTIAL ABUNDANCE  (do population sizes change
     between conditions?) and DIFFERENTIAL STATE (does signalling
     change within a population?).

All figures -> ./figures, all result tables -> ./results.

The pipeline is deliberately written to read a generic table of
markers, so the same code runs on real .fcs data once it is exported
to CSV (or loaded with fcsparser / FlowKit). See the README.
===================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from minisom import MiniSom
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from scipy.stats import mannwhitneyu
import umap

# ------------------------------------------------------------------
# Config & house style
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42
COFACTOR = 5            # standard arcsinh cofactor for fluorescence flow
SOM_DIM = 10            # 10x10 = 100 high-resolution clusters
N_METACLUSTERS = 6      # final number of populations to report

sns.set_theme(style="white", context="talk")
np.random.seed(SEED)


# ==================================================================
# STEP 1 — Load data
# ==================================================================
def load_data():
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    panel = pd.read_csv(os.path.join(DATA_DIR, "panel.csv"))
    markers = panel["marker"].tolist()
    type_markers = panel.loc[panel.marker_class == "type", "marker"].tolist()
    state_markers = panel.loc[panel.marker_class == "state", "marker"].tolist()

    frames = []
    for _, row in meta.iterrows():
        f = pd.read_csv(os.path.join(DATA_DIR, f"sample_{row.sample_id}.csv"))
        f["sample_id"] = row.sample_id
        f["patient_id"] = row.patient_id
        f["condition"] = row.condition
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    print(f"[1] Loaded {len(df):,} cells from {len(meta)} samples, "
          f"{len(markers)} markers.")
    return df, meta, markers, type_markers, state_markers


# ==================================================================
# STEP 2 — Transform (arcsinh)
# ==================================================================
def transform(df, markers):
    # arcsinh compresses the bright signals and spreads out the dim
    # ones -- the standard way to make cytometry data look "gaussian".
    df[markers] = np.arcsinh(df[markers] / COFACTOR)
    print(f"[2] Applied arcsinh transform (cofactor={COFACTOR}).")
    return df


# ==================================================================
# STEP 3 — Cluster (FlowSOM-style: SOM + hierarchical meta-clustering)
# ==================================================================
def cluster(df, type_markers):
    X = df[type_markers].to_numpy()

    # 3a. Train a Self-Organising Map -> assign each cell to 1 of 100 nodes
    som = MiniSom(SOM_DIM, SOM_DIM, X.shape[1],
                  sigma=1.0, learning_rate=0.5,
                  random_seed=SEED)
    som.random_weights_init(X)
    som.train_random(X, 20000, verbose=False)

    # node index (0..99) for every cell
    winners = np.array([som.winner(x) for x in X])
    node_id = winners[:, 0] * SOM_DIM + winners[:, 1]
    df["som_node"] = node_id

    # 3b. Meta-cluster the 100 node codebook vectors into N populations
    codebook = som.get_weights().reshape(SOM_DIM * SOM_DIM, X.shape[1])
    meta = AgglomerativeClustering(n_clusters=N_METACLUSTERS).fit_predict(codebook)
    df["cluster"] = meta[node_id]
    print(f"[3] Clustered into {SOM_DIM*SOM_DIM} SOM nodes -> "
          f"{N_METACLUSTERS} meta-clusters.")
    return df


# ==================================================================
# STEP 4 — Annotate clusters by marker profile + heatmap
# ==================================================================
def annotate(df, type_markers):
    # median expression of each type-marker within each cluster
    profile = df.groupby("cluster")[type_markers].median()

    # simple automatic labelling: name each cluster after the lineage
    # whose defining markers it expresses most strongly.
    lineage_rules = {
        "CD4 T":     ["CD4"],
        "CD8 T":     ["CD8"],
        "B cells":   ["CD19", "CD20"],
        "NK cells":  ["CD56"],
        "Monocytes": ["CD14"],
        "DC":        ["CD11c"],
    }
    # z-score markers across clusters so "high for this cluster" is comparable
    z = (profile - profile.mean()) / (profile.std() + 1e-9)
    labels = {}
    used = {}
    for cl in profile.index:
        scores = {name: z.loc[cl, mk].mean() for name, mk in lineage_rules.items()}
        best = max(scores, key=scores.get)
        # disambiguate duplicates with a numeric suffix
        used[best] = used.get(best, 0) + 1
        labels[cl] = best if used[best] == 1 else f"{best} ({used[best]})"
    df["cluster_label"] = df["cluster"].map(labels)

    # ---- heatmap figure --------------------------------------------
    profile_z = z.copy()
    profile_z.index = [labels[i] for i in profile_z.index]
    plt.figure(figsize=(11, 7))
    sns.heatmap(profile_z, cmap="RdBu_r", center=0, annot=False,
                linewidths=0.5, cbar_kws={"label": "z-scored median expr."})
    plt.title("Cluster phenotypes (z-scored marker medians)")
    plt.ylabel("Cluster"); plt.xlabel("Marker")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_cluster_heatmap.png"), dpi=150)
    plt.close()

    profile.assign(label=[labels[i] for i in profile.index]).to_csv(
        os.path.join(RES_DIR, "cluster_marker_medians.csv"))
    print(f"[4] Annotated clusters: {sorted(set(labels.values()))}")
    return df, labels


# ==================================================================
# STEP 5 — UMAP embedding for visualisation
# ==================================================================
def embed_umap(df, type_markers, n_cells=20000):
    sub = df.sample(n=min(n_cells, len(df)), random_state=SEED).copy()
    emb = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=SEED).fit_transform(
        sub[type_markers].to_numpy())
    sub["UMAP1"], sub["UMAP2"] = emb[:, 0], emb[:, 1]

    # 5a. UMAP coloured by cluster label
    plt.figure(figsize=(10, 8))
    for lab, g in sub.groupby("cluster_label"):
        plt.scatter(g.UMAP1, g.UMAP2, s=3, alpha=0.5, label=lab)
    plt.legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=11, frameon=False)
    plt.title("UMAP of immune cells, coloured by unsupervised cluster")
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_umap_clusters.png"), dpi=150)
    plt.close()

    # 5b. UMAP coloured by a few key markers (sanity check)
    key = ["CD3", "CD19", "CD14", "CD56"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, mk in zip(axes.ravel(), key):
        sc = ax.scatter(sub.UMAP1, sub.UMAP2, c=sub[mk], s=3, cmap="viridis")
        ax.set_title(mk); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("UMAP coloured by marker expression", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_umap_markers.png"), dpi=150)
    plt.close()
    print("[5] Saved UMAP figures.")


# ==================================================================
# STEP 6 — Sample-level QC: MDS plot
# ==================================================================
def mds_plot(df, meta, markers):
    # one expression "pseudobulk" vector per sample (median per marker)
    pb = df.groupby("sample_id")[markers].median()
    pb = pb.loc[meta.sample_id]
    coords = MDS(n_components=2, random_state=SEED, normalized_stress="auto"
                 ).fit_transform(pb.to_numpy())
    md = meta.copy()
    md["MDS1"], md["MDS2"] = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 7))
    for cond, g in md.groupby("condition"):
        plt.scatter(g.MDS1, g.MDS2, s=180, label=cond, alpha=0.8)
    for _, r in md.iterrows():
        plt.annotate(r.patient_id, (r.MDS1, r.MDS2), fontsize=9,
                     ha="center", va="center")
    plt.legend(title="Condition", frameon=False)
    plt.title("Sample-level MDS (pseudobulk medians)")
    plt.xlabel("MDS1"); plt.ylabel("MDS2")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_mds_samples.png"), dpi=150)
    plt.close()
    print("[6] Saved MDS QC figure.")


# ==================================================================
# STEP 7 — Differential abundance (do cluster sizes change?)
# ==================================================================
def differential_abundance(df, meta):
    counts = (df.groupby(["sample_id", "cluster_label"]).size()
                .unstack(fill_value=0))
    props = counts.div(counts.sum(axis=1), axis=0)          # proportions
    props = props.join(meta.set_index("sample_id")["condition"])

    long = props.melt(id_vars="condition", var_name="cluster",
                      value_name="proportion")

    # Wilcoxon (Mann-Whitney) test per cluster
    rows = []
    for cl, g in long.groupby("cluster"):
        a = g.loc[g.condition == "Reference", "proportion"]
        b = g.loc[g.condition == "BCRXL", "proportion"]
        stat, p = mannwhitneyu(a, b)
        rows.append(dict(cluster=cl, ref_mean=a.mean(),
                         bcrxl_mean=b.mean(), p_value=p))
    da = pd.DataFrame(rows).sort_values("p_value")
    da.to_csv(os.path.join(RES_DIR, "differential_abundance.csv"), index=False)

    plt.figure(figsize=(13, 7))
    sns.boxplot(data=long, x="cluster", y="proportion", hue="condition")
    plt.xticks(rotation=40, ha="right")
    plt.title("Differential abundance: cluster proportions by condition")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_differential_abundance.png"), dpi=150)
    plt.close()
    print("[7] Differential abundance done. Top hit:")
    print(da.head(1).to_string(index=False))
    return da


# ==================================================================
# STEP 8 — Differential state (does signalling change within B cells?)
# ==================================================================
def differential_state(df, meta, state_markers):
    rows = []
    for mk in state_markers:
        med = (df.groupby(["sample_id", "cluster_label"])[mk]
                 .median().reset_index())
        med = med.merge(meta[["sample_id", "condition"]], on="sample_id")
        for cl, g in med.groupby("cluster_label"):
            a = g.loc[g.condition == "Reference", mk]
            b = g.loc[g.condition == "BCRXL", mk]
            if len(a) > 1 and len(b) > 1:
                stat, p = mannwhitneyu(a, b)
                rows.append(dict(marker=mk, cluster=cl,
                                 ref_median=a.median(),
                                 bcrxl_median=b.median(), p_value=p))
    ds = pd.DataFrame(rows).sort_values("p_value")
    ds.to_csv(os.path.join(RES_DIR, "differential_state.csv"), index=False)

    # focused plot on the headline result: pS6 in B cells
    bcell_label = [c for c in df.cluster_label.unique() if c.startswith("B cells")]
    if bcell_label:
        bc = bcell_label[0]
        med = (df[df.cluster_label == bc]
               .groupby(["sample_id"])["pS6"].median().reset_index())
        med = med.merge(meta[["sample_id", "condition"]], on="sample_id")
        plt.figure(figsize=(7, 7))
        sns.boxplot(data=med, x="condition", y="pS6",
                    order=["Reference", "BCRXL"])
        sns.stripplot(data=med, x="condition", y="pS6",
                      order=["Reference", "BCRXL"], color="black", size=8)
        plt.title(f"Differential state: pS6 in {bc}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "06_pS6_in_Bcells.png"), dpi=150)
        plt.close()
    print("[8] Differential state done. Top hit:")
    print(ds.head(1).to_string(index=False))
    return ds


# ==================================================================
# MAIN
# ==================================================================
def main():
    df, meta, markers, type_markers, state_markers = load_data()
    df = transform(df, markers)
    df = cluster(df, type_markers)
    df, labels = annotate(df, type_markers)
    embed_umap(df, type_markers)
    mds_plot(df, meta, markers)
    differential_abundance(df, meta)
    differential_state(df, meta, state_markers)

    # quick check: how well did unsupervised clusters recover truth?
    if "true_label" in df.columns:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(df["true_label"], df["cluster"])
        print(f"\n[QC] Adjusted Rand Index vs ground truth: {ari:.3f} "
              f"(1.0 = perfect)")
    print("\nDone. See ./figures and ./results.")


if __name__ == "__main__":
    main()
