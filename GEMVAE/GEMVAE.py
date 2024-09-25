import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from .model import GATE
from .utils import build_adjacency


class GEMVAE:
    """
    Graph-Enhanced Multi-modal Variational Autoencoder (GemVAE).

    Combines a dual-stream GATE encoder with spatial graph attention,
    variational bottleneck, and contrastive spatial coherence loss
    for joint transcriptomic-proteomic spatial embedding.

    Parameters
    ----------
    hidden_dims1    Hidden dimensions for gene encoder, e.g. [512, 256].
    hidden_dims2    Hidden dimensions for protein encoder, e.g. [128, 64].
    z_dim           Latent dimension (default 30).
    n_epochs        Training epochs (default 500).
    lr              Adam learning rate (default 1e-4).
    weight_decay    L2 regularization coefficient (default 1e-4).
    grad_clip       Gradient clipping norm (default 5).
    lambda_recon    Reconstruction loss weight (default 1).
    lambda_kl       KL divergence weight (default 0 = deterministic).
    lambda_contrast Contrastive loss weight (default 10).
    tau             Contrastive temperature (default 0.5).
    alpha_blend     Blend ratio of full vs pruned adjacency in decoder (default 0.5).
    random_seed     Reproducibility seed (default 2020).
    """

    def __init__(
        self,
        hidden_dims1=(512, 256),
        hidden_dims2=(128, 64),
        z_dim=30,
        n_epochs=500,
        lr=1e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
        lambda_recon=1.0,
        lambda_kl=0.0,
        lambda_contrast=10.0,
        tau=0.5,
        alpha_blend=0.5,
        random_seed=2020,
    ):
        self.hidden_dims1    = list(hidden_dims1)
        self.hidden_dims2    = list(hidden_dims2)
        self.z_dim           = z_dim
        self.n_epochs        = n_epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.grad_clip       = grad_clip
        self.lambda_recon    = lambda_recon
        self.lambda_kl       = lambda_kl
        self.lambda_contrast = lambda_contrast
        self.tau             = tau
        self.alpha_blend     = alpha_blend
        self.random_seed     = random_seed
        self._built          = False

    # ------------------------------------------------------------------
    def _sparse_to_tf(self, A_sp):
        """Convert a scipy sparse matrix to a TF SparseTensor."""
        A_coo = A_sp.tocoo().astype(np.float32)
        indices = np.column_stack([A_coo.row, A_coo.col]).astype(np.int64)
        values  = A_coo.data
        shape   = A_coo.shape
        return tf.SparseTensor(indices=indices, values=values,
                               dense_shape=[shape[0], shape[1]])

    # ------------------------------------------------------------------
    def _contrastive_loss(self, Z, A_sparse_dense):
        """NT-Xent contrastive loss over spatial neighbors."""
        Z_norm = tf.nn.l2_normalize(Z, axis=-1)
        sim    = tf.matmul(Z_norm, Z_norm, transpose_b=True) / self.tau
        exp_sim = tf.exp(sim)

        pos_mask = A_sparse_dense  # [N, N], 1 if neighbors
        pos_sum  = tf.reduce_sum(exp_sim * pos_mask, axis=-1)
        all_sum  = tf.reduce_sum(exp_sim, axis=-1) - tf.linalg.diag_part(exp_sim)
        loss     = -tf.reduce_mean(tf.math.log(pos_sum / (all_sum + 1e-8) + 1e-8))
        return loss

    # ------------------------------------------------------------------
    def _kl_loss(self, mu, log_var):
        return -0.5 * tf.reduce_mean(1.0 + log_var - tf.square(mu) - tf.exp(log_var))

    # ------------------------------------------------------------------
    def train(self, adata, X_gene, X_prot, verbose=True):
        """
        Train GemVAE on the provided multi-omic data.

        Parameters
        ----------
        adata   AnnData with adata.uns['Spatial_Net'] already set.
        X_gene  np.ndarray [N, d_gene], normalized gene expression.
        X_prot  np.ndarray [N, d_protein], normalized protein counts.
        verbose Print loss every 50 epochs.
        """
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        A_norm  = build_adjacency(adata)
        A_prune = self._prune_adjacency(A_norm, percentile=80)

        A_tf        = self._sparse_to_tf(A_norm)
        A_prune_tf  = self._sparse_to_tf(A_prune)
        A_dense     = tf.sparse.to_dense(tf.sparse.reorder(A_tf))

        X_g = tf.constant(X_gene.astype(np.float32))
        X_p = tf.constant(X_prot.astype(np.float32))

        gate = GATE(self.hidden_dims1, self.hidden_dims2, self.z_dim)

        # Initialize weights by running one forward pass
        gate._init_weights(X_gene.shape[1], X_prot.shape[1])

        optimizer = tf.optimizers.Adam(learning_rate=self.lr)

        all_vars = list(gate.weights.values())

        for epoch in range(1, self.n_epochs + 1):
            with tf.GradientTape() as tape:
                z, mu, log_var, X_g_hat, X_p_hat = gate.forward(
                    X_g, X_p, A_tf, A_prune_tf, training=True, use_vae=(self.lambda_kl > 0)
                )

                loss_recon = (
                    tf.sqrt(tf.reduce_sum(tf.square(X_g - X_g_hat))) +
                    tf.sqrt(tf.reduce_sum(tf.square(X_p - X_p_hat)))
                )
                loss_kl       = self._kl_loss(mu, log_var) if self.lambda_kl > 0 else 0.0
                loss_contrast = self._contrastive_loss(z, A_dense)
                loss_wd       = tf.add_n([tf.nn.l2_loss(v) for v in all_vars]) * self.weight_decay

                total_loss = (
                    self.lambda_recon    * loss_recon +
                    self.lambda_kl       * loss_kl +
                    self.lambda_contrast * loss_contrast +
                    loss_wd
                )

            grads = tape.gradient(total_loss, all_vars)
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            optimizer.apply_gradients(zip(grads, all_vars))

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:4d}/{self.n_epochs} | "
                      f"Loss: {total_loss.numpy():.4f} | "
                      f"Recon: {loss_recon.numpy():.4f} | "
                      f"Contrast: {loss_contrast.numpy():.4f}")

        self._gate = gate
        self._A_tf = A_tf
        self._A_prune_tf = A_prune_tf
        return self

    # ------------------------------------------------------------------
    def get_embeddings(self, X_gene, X_prot):
        """Return latent embeddings (mu) after training."""
        if not hasattr(self, "_gate"):
            raise RuntimeError("Call train() before get_embeddings().")
        X_g = tf.constant(X_gene.astype(np.float32))
        X_p = tf.constant(X_prot.astype(np.float32))
        _, mu, _, _, _ = self._gate.forward(
            X_g, X_p, self._A_tf, self._A_prune_tf, training=False, use_vae=False
        )
        return mu.numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def _prune_adjacency(A_sp, percentile=80):
        """Remove weak edges below a distance percentile threshold."""
        A_coo = A_sp.tocoo()
        threshold = np.percentile(A_coo.data[A_coo.data > 0], percentile)
        mask = A_coo.data >= threshold
        return sp.coo_matrix(
            (A_coo.data[mask], (A_coo.row[mask], A_coo.col[mask])),
            shape=A_sp.shape, dtype=np.float32
        ).tocsr()
