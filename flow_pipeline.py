"""
flow_pipeline.py
================
Reusable building blocks for an unsupervised flow cytometry analysis.

The same five steps are used for both the simulated dataset (where we know the
true cell types) and the real .fcs file (where we don't). Keeping them in one
module means the two analyses are guaranteed to use *identical* logic.

Pipeline steps
--------------
1. load        -> get an (n_cells x n_markers) expression table
2. transform   -> compress the huge fluorescence dynamic range (arcsinh)
3. scale       -> z-score each marker so none dominates the distance metric
4. cluster     -> FlowSOM-style: train a 10x10 SOM, then metacluster its nodes
5. embed       -> UMAP, only for 2-D visualisation (never for clustering itself)

Why FlowSOM-style clustering?
-----------------------------
This mirrors the standard CATALYST / FlowSOM workflow used across the field:
a Self-Organizing Map first compresses millions of cells into 100 prototype
nodes, then those nodes are merged ("metaclustered") into a handful of
interpretable populations. It is fast, reproducible, and is the de-facto
benchmark winner for high-dimensional cytometry clustering
(Weber & Robinson, Cytometry A 2016).
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------
def load_fcs(path: str, compensate: bool = True):
    """Read a real .fcs file and return a tidy DataFrame of marker expression.

    Parameters
    ----------
    path : str
        Path to the .fcs file.
    compensate : bool
        Apply the spillover/compensation matrix stored in the file, if present.
        Compensation corrects for the fact that each fluorophore "spills" a bit
        of its signal into neighbouring detectors.

    Returns
    -------
    df : pd.DataFrame  (n_cells x n_channels), columns named by marker.
    """
    import flowkit as fk

    sample = fk.Sample(path)

    # Apply the embedded compensation matrix if one exists.
    if compensate and "spill" in sample.metadata:
        sample.apply_compensation(sample.metadata["spill"])
        source = "comp"
    else:
        source = "raw"

    df = sample.as_dataframe(source=source)
    # FlowKit gives a 2-level column index (detector, marker); prefer the
    # human-readable marker name when present, else fall back to the detector.
    df.columns = [m if m else d for d, m in df.columns]
    return df


# ----------------------------------------------------------------------
# 2. TRANSFORM
# ----------------------------------------------------------------------
def arcsinh_transform(df: pd.DataFrame, markers: list[str], cofactor: float = 150.0):
    """Arcsinh-transform the chosen marker columns.

    Raw flow data spans ~0 to 260,000. The arcsinh transform behaves linearly
    near zero (so negative/background populations stay sensible) and
    logarithmically for large values (so bright populations are compressed).
    cofactor=150 is a common default for fluorescence-based flow cytometry.
    """
    X = df[markers].to_numpy(dtype=float)
    Xt = np.arcsinh(X / cofactor)
    return pd.DataFrame(Xt, columns=markers, index=df.index)


# ----------------------------------------------------------------------
# 3. SCALE
# ----------------------------------------------------------------------
def zscore(df: pd.DataFrame):
    """Standardise each marker to mean 0 / std 1.

    Without this, a marker measured on a wider numeric range would dominate the
    Euclidean distance used by the SOM, biasing the clustering.
    """
    X = df.to_numpy(dtype=float)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return pd.DataFrame(Xz, columns=df.columns, index=df.index)


# ----------------------------------------------------------------------
# 4. CLUSTER  (FlowSOM-style)
# ----------------------------------------------------------------------
def flowsom_cluster(Xz: np.ndarray, n_metaclusters: int = 10,
                    grid: int = 10, iters: int = 30000, seed: int = 0):
    """Two-step FlowSOM-style clustering.

    Step 1: train a (grid x grid) Self-Organizing Map -> grid*grid prototype nodes.
    Step 2: hierarchically merge those nodes into `n_metaclusters` populations.

    Returns one integer cluster label per cell.
    """
    from minisom import MiniSom
    from sklearn.cluster import AgglomerativeClustering

    n_nodes = grid * grid

    som = MiniSom(grid, grid, Xz.shape[1],
                  sigma=1.0, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(Xz)
    som.train_random(Xz, iters)

    # Which SOM node does each cell map to? -> flatten (row, col) into 0..n_nodes-1
    win = np.array([som.winner(x) for x in Xz])
    node_of_cell = win[:, 0] * grid + win[:, 1]

    # The codebook: one prototype vector per node. Metacluster these nodes.
    codebook = som.get_weights().reshape(n_nodes, Xz.shape[1])
    meta = AgglomerativeClustering(n_clusters=n_metaclusters).fit_predict(codebook)

    return meta[node_of_cell]


# ----------------------------------------------------------------------
# 5. EMBED  (visualisation only)
# ----------------------------------------------------------------------
def umap_embed(Xz: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.2,
               seed: int = 0, max_cells: int = 20000):
    """Return a 2-D UMAP embedding for plotting.

    For speed/readability we optionally subsample to `max_cells`. UMAP is used
    ONLY to *look* at the data; clustering is always done on the full-D space.

    Returns (embedding 2-D array, index of cells used).
    """
    import umap

    n = Xz.shape[0]
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n, max_cells, replace=False) if n > max_cells
           else np.arange(n))
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                    random_state=seed).fit_transform(Xz[idx])
    return emb, idx


# ----------------------------------------------------------------------
# ANNOTATION HELPERS
# ----------------------------------------------------------------------
def cluster_marker_medians(expr: pd.DataFrame, clusters: np.ndarray):
    """Median expression of each marker within each cluster (a 'phenotype' table)."""
    tbl = expr.copy()
    tbl["cluster"] = clusters
    return tbl.groupby("cluster").median()


def cluster_frequencies(clusters: np.ndarray):
    """Percentage of cells in each cluster."""
    s = pd.Series(clusters).value_counts().sort_index()
    return (100 * s / s.sum()).round(2)
