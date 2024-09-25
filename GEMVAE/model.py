import tensorflow as tf
import numpy as np


def _linbndrop(x, w, b, use_bn=True, dropout_rate=0.2, activation=None, training=False):
    """BatchNorm -> Dropout -> Linear block."""
    if use_bn:
        x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.dropout(x, rate=dropout_rate if training else 0.0)
    x = tf.matmul(x, w) + b
    if activation is not None:
        x = activation(x)
    return x


class GATE:
    """
    Graph ATtention Encoder-decoder for multi-modal spatial omics.

    Dual-stream architecture: independent encoders for gene expression
    (hidden_dims1) and protein abundance (hidden_dims2), followed by
    joint graph attention layers and a variational bottleneck.
    """

    def __init__(self, hidden_dims1, hidden_dims2, z_dim=30):
        self.hidden_dims1 = hidden_dims1   # e.g. [512, 256]
        self.hidden_dims2 = hidden_dims2   # e.g. [128, 64]
        self.z_dim = z_dim
        self.weights = {}
        self.attention = {}

    def _init_weights(self, d_gene, d_protein):
        W = self.weights

        # ----- Gene encoder -----
        dims1 = [d_gene] + self.hidden_dims1
        for i in range(len(dims1) - 1):
            W[f"W_g{i}"] = tf.Variable(
                tf.glorot_uniform_initializer()([dims1[i], dims1[i + 1]]),
                name=f"W_g{i}", dtype=tf.float32
            )
            W[f"b_g{i}"] = tf.Variable(tf.zeros([dims1[i + 1]]), name=f"b_g{i}")

        # ----- Protein encoder -----
        dims2 = [d_protein] + self.hidden_dims2
        for i in range(len(dims2) - 1):
            W[f"W_p{i}"] = tf.Variable(
                tf.glorot_uniform_initializer()([dims2[i], dims2[i + 1]]),
                name=f"W_p{i}", dtype=tf.float32
            )
            W[f"b_p{i}"] = tf.Variable(tf.zeros([dims2[i + 1]]), name=f"b_p{i}")

        # ----- Joint projection -----
        joint_in = self.hidden_dims1[-1] + self.hidden_dims2[-1]
        joint_out = self.hidden_dims1[-1]  # project back to gene encoder dim
        W["W_joint"] = tf.Variable(
            tf.glorot_uniform_initializer()([joint_in, joint_out]), name="W_joint"
        )
        W["b_joint"] = tf.Variable(tf.zeros([joint_out]), name="b_joint")

        # ----- Graph attention: dual paths -----
        h = joint_out
        W["W_att"] = tf.Variable(
            tf.glorot_uniform_initializer()([h, h]), name="W_att"
        )
        W["a1"] = tf.Variable(tf.random.normal([2 * h, 1]), name="a1")
        W["a2"] = tf.Variable(tf.random.normal([2 * h, 1]), name="a2")

        # ----- VAE bottleneck -----
        W["W_mu"]  = tf.Variable(tf.glorot_uniform_initializer()([h, self.z_dim]), name="W_mu")
        W["b_mu"]  = tf.Variable(tf.zeros([self.z_dim]), name="b_mu")
        W["W_var"] = tf.Variable(tf.glorot_uniform_initializer()([h, self.z_dim]), name="W_var")
        W["b_var"] = tf.Variable(tf.zeros([self.z_dim]), name="b_var")

        # ----- Decoder: gene -----
        W["W_dec_g"] = tf.Variable(
            tf.glorot_uniform_initializer()([self.z_dim, d_gene]), name="W_dec_g"
        )
        W["b_dec_g"] = tf.Variable(tf.zeros([d_gene]), name="b_dec_g")

        # ----- Decoder: protein -----
        W["W_dec_p"] = tf.Variable(
            tf.glorot_uniform_initializer()([self.z_dim, d_protein]), name="W_dec_p"
        )
        W["b_dec_p"] = tf.Variable(tf.zeros([d_protein]), name="b_dec_p")

    def _encode_modalities(self, X_gene, X_prot, training=False):
        """Dual-stream modality-specific encoding."""
        # Gene stream
        h_g = X_gene
        for i in range(len(self.hidden_dims1)):
            h_g = _linbndrop(
                h_g, self.weights[f"W_g{i}"], self.weights[f"b_g{i}"],
                activation=tf.nn.elu, training=training
            )

        # Protein stream
        h_p = X_prot
        for i in range(len(self.hidden_dims2)):
            h_p = _linbndrop(
                h_p, self.weights[f"W_p{i}"], self.weights[f"b_p{i}"],
                activation=tf.nn.elu, training=training
            )

        # Joint projection
        h = tf.concat([h_g, h_p], axis=-1)
        h = tf.nn.elu(tf.matmul(h, self.weights["W_joint"]) + self.weights["b_joint"])
        return h

    def _graph_attention(self, H, A_sparse):
        """
        Bidirectional dual-path graph attention.
        H: [N, h], A_sparse: SparseTensor [N, N]
        Returns attended H': [N, h]
        """
        W = self.weights["W_att"]
        Wh = tf.matmul(H, W)  # [N, h]

        # Compute pairwise attention logits over sparse edges
        N = tf.shape(H)[0]
        idx = A_sparse.indices  # [E, 2]
        Wh_i = tf.gather(Wh, idx[:, 0])  # [E, h]
        Wh_j = tf.gather(Wh, idx[:, 1])  # [E, h]
        cat = tf.concat([Wh_i, Wh_j], axis=-1)  # [E, 2h]

        e1 = tf.squeeze(tf.matmul(cat, self.weights["a1"]), axis=-1)  # [E]
        e2 = tf.squeeze(tf.matmul(cat, self.weights["a2"]), axis=-1)  # [E]
        e  = tf.sigmoid(e1 + e2)

        # Softmax normalization per source node
        alpha = tf.SparseTensor(indices=idx, values=e, dense_shape=tf.cast([N, N], tf.int64))
        alpha = tf.sparse.softmax(alpha)
        alpha_values = alpha.values

        # Aggregate: H'[i] = sum_j alpha_ij * Wh_j
        weighted = alpha_values[:, tf.newaxis] * Wh_j  # [E, h]
        H_prime  = tf.math.unsorted_segment_sum(weighted, idx[:, 0], num_segments=N)
        return H_prime

    def forward(self, X_gene, X_prot, A_sparse, A_pruned_sparse, training=False, use_vae=False):
        """Full encoder-decoder forward pass."""
        if not self.weights:
            self._init_weights(
                X_gene.shape[-1].value or tf.shape(X_gene)[-1],
                X_prot.shape[-1].value or tf.shape(X_prot)[-1]
            )

        # Encode
        H = self._encode_modalities(X_gene, X_prot, training)

        # Graph attention (encoder)
        H_prime = self._graph_attention(H, A_sparse)

        # Variational bottleneck
        mu  = tf.matmul(H_prime, self.weights["W_mu"])  + self.weights["b_mu"]
        var = tf.matmul(H_prime, self.weights["W_var"]) + self.weights["b_var"]

        if use_vae:
            eps = tf.random.normal(tf.shape(mu))
            z   = mu + eps * tf.exp(0.5 * var)
        else:
            z = mu

        # Decode
        X_gene_hat = tf.matmul(z, self.weights["W_dec_g"]) + self.weights["b_dec_g"]
        X_prot_hat = tf.matmul(z, self.weights["W_dec_p"]) + self.weights["b_dec_p"]

        self.attention["mu"]  = mu
        self.attention["var"] = var
        self.attention["z"]   = z

        return z, mu, var, X_gene_hat, X_prot_hat
