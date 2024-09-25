import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model="Radius", verbose=True):
    """
    Build a 2-D spatial graph from tissue coordinates.

    Parameters
    ----------
    adata       AnnData with adata.obsm['spatial'] coordinates.
    rad_cutoff  Radius threshold (used when model='Radius').
    k_cutoff    Number of nearest neighbors (used when model='KNN').
    model       'Radius' or 'KNN'.
    verbose     Print edge statistics.
    """
    assert model in ("Radius", "KNN"), "model must be 'Radius' or 'KNN'"
    coords = np.array(adata.obsm["spatial"])
    cell_ids = adata.obs.index.tolist()

    if model == "KNN":
        nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
    else:
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(coords)
        distances, indices = nbrs.radius_neighbors(coords)

    rows, cols, dists = [], [], []
    for i, (idx_row, dist_row) in enumerate(zip(indices, distances)):
        for j, d in zip(idx_row, dist_row):
            if d > 0:  # exclude self-loops
                rows.append(cell_ids[i])
                cols.append(cell_ids[j])
                dists.append(d)

    net = pd.DataFrame({"Cell1": rows, "Cell2": cols, "Distance": dists})
    adata.uns["Spatial_Net"] = net

    if verbose:
        print(f"The Spatial_Net has {net.shape[0]} edges, "
              f"mean {net.shape[0] / len(cell_ids):.1f} neighbors per cell.")
    return adata


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis, key_section="Section_id", verbose=True):
    """
    Build a 3-D spatial graph across multiple tissue sections.

    Within each section: radius graph with rad_cutoff_2D.
    Between adjacent sections: radius graph with rad_cutoff_Zaxis.
    """
    assert key_section in adata.obs.columns, f"'{key_section}' column not in adata.obs"
    sections = adata.obs[key_section].unique().tolist()
    all_nets = []

    # 2-D intra-section graphs
    for sec in sections:
        idx = adata.obs[key_section] == sec
        sub = adata[idx].copy()
        Cal_Spatial_Net(sub, rad_cutoff=rad_cutoff_2D, model="Radius", verbose=False)
        net = sub.uns["Spatial_Net"].copy()
        net["Section_id"] = sec
        all_nets.append(net)

    # Z-axis inter-section edges
    coords = np.array(adata.obsm["spatial"])
    cell_ids = adata.obs.index.tolist()
    sec_labels = adata.obs[key_section].values

    for k in range(len(sections) - 1):
        mask_a = sec_labels == sections[k]
        mask_b = sec_labels == sections[k + 1]
        ids_a  = [cell_ids[i] for i, m in enumerate(mask_a) if m]
        ids_b  = [cell_ids[i] for i, m in enumerate(mask_b) if m]
        pts_a  = coords[mask_a]
        pts_b  = coords[mask_b]

        nbrs = NearestNeighbors(radius=rad_cutoff_Zaxis).fit(pts_b)
        dists_ab, inds_ab = nbrs.radius_neighbors(pts_a)

        rows, cols, d_vals = [], [], []
        for i, (idx_row, dist_row) in enumerate(zip(inds_ab, dists_ab)):
            for j, d in zip(idx_row, dist_row):
                rows.append(ids_a[i])
                cols.append(ids_b[j])
                d_vals.append(d)

        if rows:
            z_net = pd.DataFrame({"Cell1": rows, "Cell2": cols, "Distance": d_vals,
                                  "Section_id": f"{sections[k]}-{sections[k+1]}", "SNN": True})
            all_nets.append(z_net)

    net_combined = pd.concat(all_nets, ignore_index=True)
    adata.uns["Spatial_Net"] = net_combined

    if verbose:
        print(f"3-D Spatial_Net: {net_combined.shape[0]} total edges across {len(sections)} sections.")
    return adata


def Stats_Spatial_Net(adata):
    """Plot neighbor-count distribution of the spatial graph."""
    net = adata.uns["Spatial_Net"]
    n_cells = adata.n_obs
    counts = net.groupby("Cell1").size()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.value_counts().sort_index().index,
           counts.value_counts().sort_index().values / n_cells * 100)
    ax.set_xlabel("Number of spatial neighbors")
    ax.set_ylabel("Percentage of cells (%)")
    ax.set_title("Spatial network neighbor distribution")
    plt.tight_layout()
    plt.show()
    print(f"Total edges: {net.shape[0]}  |  Mean neighbors: {net.shape[0] / n_cells:.2f}")


def build_adjacency(adata):
    """Convert Spatial_Net DataFrame to a normalized sparse adjacency matrix."""
    net   = adata.uns["Spatial_Net"]
    cells = adata.obs.index.tolist()
    idx   = {c: i for i, c in enumerate(cells)}
    N     = len(cells)

    rows = [idx[c] for c in net["Cell1"] if c in idx]
    cols = [idx[c] for c in net["Cell2"] if c in idx]

    A = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(N, N), dtype=np.float32
    ).tocsr()

    # Symmetric + self-loops
    A = A + A.T + sp.eye(N, dtype=np.float32)

    # Row-normalize
    d = np.array(A.sum(axis=1)).flatten()
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    D_inv = sp.diags(d_inv)
    A_norm = D_inv.dot(A)
    return A_norm
