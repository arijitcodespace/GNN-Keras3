from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, activations, initializers

try:
    from .base import GraphLayer
    from ..ops import add_self_loops
except Exception:
    from src.gnn_keras.layers.base import GraphLayer
    from src.gnn_keras.ops import add_self_loops

class GraphAttention(GraphLayer):
    """GAT layer skeleton (Veličković et al.).

    Efficient on sparse graphs via `tf.sparse`. For dense graphs, uses
    masked attention over the adjacency.
    """
    def __init__(self, units: int, heads: int = 1, concat: bool = True,
                 dropout: float = 0.0, activation = "elu", self_loops = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.self_loops = self_loops
        # Linear projection for features per head
        self.lin = layers.Dense(units * heads, use_bias = False)
        self.a_src = self.add_weight(shape = (heads, units), initializer = initializers.GlorotUniform(), name = "a_src")
        self.a_dst = self.add_weight(shape = (heads, units), initializer = initializers.GlorotUniform(), name = "a_dst")
        self.bias = self.add_weight(shape = (units * heads,), initializer = "zeros", name = "bias")
        self.dropout_layer = layers.Dropout(dropout)

    def _activation_and_combine(self, h):
        h = h + self.bias
        if self.activation is not None:
            h = self.activation(h)
        if not self.concat and self.heads > 1:
            # average over heads
            h = tf.reduce_mean(tf.reshape(h, [-1, self.heads, self.units]), axis = 1)
        return h

    def _call_dense(self, x, a, training = None):
        if self.self_loops:
            a = add_self_loops(a)
        N = tf.shape(x)[0]
        H, U = self.heads, self.units
        h = self.lin(x)  # [N, units*H]
        h = tf.reshape(h, [N, H, U])  # [N,H,U]
        # compute attention scores e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e_src = tf.einsum("nhu,hu->nh", h, self.a_src)  # [N,H]
        e_dst = tf.einsum("nhu,hu->nh", h, self.a_dst)  # [N,H]
        logits = tf.expand_dims(e_src, 1) + tf.expand_dims(e_dst, 0)  # [N,N,H]
        logits = tf.nn.leaky_relu(logits, alpha = 0.2)

        # mask with adjacency and softmax per node i
        mask = tf.cast(a > 0, tf.bool)
        neg_inf = tf.constant(-1e9, dtype = logits.dtype)
        logits = tf.where(tf.expand_dims(mask, -1), logits, neg_inf)
        attn = tf.nn.softmax(logits, axis = 1)  # neighbor softmax per head
        attn = self.dropout_layer(attn, training = training)

        # aggregate: out[i,h,:] = sum_j attn[i,j,h] * h[j,h,:]
        out = tf.einsum("ijh,jhu->ihu", attn, h)  # [N,H,U]
        out = tf.reshape(out, [N, H * U])
        return self._activation_and_combine(out)

    def _call_sparse(self, x, a: tf.SparseTensor, training = None):
        if self.self_loops:
            a = add_self_loops(a)
        N = tf.shape(x)[0]
        H, U = self.heads, self.units
        h = self.lin(x)  # [N, U*H]
        h = tf.reshape(h, [N, H, U])  # [N,H,U]
        e_src = tf.einsum("nhu,hu->nh", h, self.a_src)  # [N,H]
        e_dst = tf.einsum("nhu,hu->nh", h, self.a_dst)  # [N,H]

        # edges i->j
        row = a.indices[:, 0]  # [E]
        col = a.indices[:, 1]  # [E]
        logits = tf.gather(e_src, row) + tf.gather(e_dst, col)  # [E,H]
        logits = tf.nn.leaky_relu(logits, alpha = 0.2)

        # per-node (row) softmax for each head
        max_per_row = tf.math.unsorted_segment_max(logits, row, N)       # [N, H]
        logits = logits - tf.gather(max_per_row, row)                    # [E, H] stabilize
        exp = tf.exp(logits)
        denom = tf.math.unsorted_segment_sum(exp, row, N)                # [N, H]
        attn = exp / (tf.gather(denom, row) + 1e-9)                      # [E, H]

        # (optional) edge dropout on attention coefficients
        attn = self.dropout_layer(attn, training = training)

        # message passing: sum_j (alpha_ij * Wh_j)
        h_j = tf.gather(h, col)  # [E,H,U]
        m = h_j * tf.expand_dims(attn, -1)  # [E,H,U]
        out = tf.math.unsorted_segment_sum(m, row, N)  # [N,H,U]
        out = tf.reshape(out, [N, H * U])
        return self._activation_and_combine(out)
