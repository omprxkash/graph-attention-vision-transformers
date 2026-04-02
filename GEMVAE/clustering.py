import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def mclust_R(adata, num_cluster, use_rep="GemVAE", key_added="mclust", modelNames="EEE", random_seed=2020):
    """
    Gaussian mixture model clustering via R's mclust package.

    Requires rpy2 and the R mclust package to be installed.

    Parameters
    ----------
    adata        AnnData whose obsm[use_rep] contains latent embeddings.
    num_cluster  Number of clusters (Gaussian components).
    use_rep      Key in adata.obsm for the embedding.
    key_added    Key in adata.obs to store cluster labels.
    modelNames   mclust model type (default 'EEE' = equal covariance).
    random_seed  Random seed passed to R set.seed().
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        robjects.r["set.seed"](random_seed)

        r_mclust = robjects.r
        r_mclust["library"]("mclust")

        embedding = adata.obsm[use_rep]
        r_embedding = numpy2ri.numpy2rpy(embedding)
        res = r_mclust["Mclust"](r_embedding, G=num_cluster, modelNames=modelNames)
        labels = np.array(r_mclust["$"](res, "classification")).astype(str)
        adata.obs[key_added] = pd.Categorical(labels)
        adata.obs[key_added] = adata.obs[key_added].astype("category")
    except ImportError:
        raise ImportError(
            "rpy2 is required for mclust_R. "
            "Install it with 'pip install rpy2' and ensure R + mclust package are available."
        )
    return adata


def leiden_clustering(adata, use_rep="GemVAE", key_added="leiden", resolution=1.0, n_neighbors=15, random_seed=2020):
    """
    Leiden community detection on a kNN graph built from the latent embedding.

    Parameters
    ----------
    adata       AnnData with obsm[use_rep].
    use_rep     Latent embedding key.
    key_added   Column name for cluster labels in adata.obs.
    resolution  Leiden resolution parameter.
    n_neighbors Number of neighbors for the kNN graph.
    random_seed Random seed.
    """
    import scanpy as sc
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_seed, key_added=key_added)
    return adata


def louvain_clustering(adata, use_rep="GemVAE", key_added="louvain", resolution=1.0, n_neighbors=15, random_seed=2020):
    """
    Louvain community detection on a kNN graph built from the latent embedding.
    """
    import scanpy as sc
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_seed)
    sc.tl.louvain(adata, resolution=resolution, random_state=random_seed, key_added=key_added)
    return adata
