from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, activations

try:
    from ..base import GraphLayer
except Exception:
    from src.gnn_keras.layers.base import GraphLayer

class DiffPool(GraphLayer):
    """Differentiable pooling layer skeleton.

    Given node features X and adjacency A, learns an assignment S and produces
    coarsened features X' = S^T X and adjacency A' = S^T A S.
    This layer returns an updated Graph with coarsened (pooled) features/adj.
    """
    def __init__(self, clusters: int, activation = "softmax", **kwargs):
        super().__init__(**kwargs)
        self.clusters = clusters
        self.act = activations.get(activation)
        self.proj = layers.Dense(clusters)  # logits for assignment

    def _pool_dense(self, x, a):
        S = self.act(self.proj(x))  # [N,C]
        Xp = tf.linalg.matmul(S, x, transpose_a = True)  # [C,F]
        Ap = tf.linalg.matmul(tf.linalg.matmul(S, a, transpose_a = True), S)  # [C,C]
        return Xp, Ap

    def _pool_sparse(self, x, a: tf.SparseTensor):
        S = self.act(self.proj(x))  # [N,C]
        Xp = tf.linalg.matmul(S, x, transpose_a = True)  # [C,F]
        AS = tf.sparse.sparse_dense_matmul(a, S)  # [N,C]
        Ap = tf.linalg.matmul(S, AS, transpose_a = True)  # [C,C]
        return Xp, Ap

    def _call_dense(self, x, a, training = None):
        Xp, Ap = self._pool_dense(x, a)
        return x * 0 + Xp  # replaced via GraphLayer to set x; A updated in with_ below

    def _call_sparse(self, x, a, training = None):
        Xp, Ap = self._pool_sparse(x, a)
        return x * 0 + Xp

    def call(self, graph, training = None):
        x, a = graph.x, graph.a
        if isinstance(a, tf.SparseTensor):
            Xp, Ap = self._pool_sparse(x, a)
        else:
            Xp, Ap = self._pool_dense(x, a)
        return graph.with_(x = Xp, a = Ap)
    
# --------------------------------------------------------------------
# Batched, segment-aware DiffPool (K clusters PER graph, column-offset)
# --------------------------------------------------------------------
class BatchedDiffPool(layers.Layer):
    """Segment-aware DiffPool that replicates K clusters per graph.

    It reuses the Dense->K projection from a provided DiffPool instance (if any),
    applies the activation (softmax by default) row-wise to get local S,
    and constructs a block-diagonal assignment S_bd by **offsetting columns**
    for each graph in the batch.

    Vectorized build (no tf.map_fn), so variable graph sizes are fine.
    """
    def __init__(self, clusters: int, diffpool: DiffPool | None = None, activation: str = "softmax"):
        super().__init__()
        self.K = int(clusters)
        if diffpool is not None and hasattr(diffpool, "proj"):
            self.proj = diffpool.proj
            self.act = diffpool.act
        else:
            self.proj = layers.Dense(self.K)
            self.act = activations.get(activation)

    def build_S_blockdiag(self, Z: tf.Tensor, n_nodes: tf.Tensor):
        """
        Z: [N_tot, F]
        n_nodes: [B] int32 — number of nodes per graph in the packed batch.

        Returns:
            Sbd_dense: [N_tot, K*B] dense assignment matrix with column offsets,
            S_local:   [N_tot, K]   per-node assignment (no offsets),
            seg_ids:   [K*B]        segment ids (graph id per pooled node)
        """
        K = tf.constant(self.K, dtype = tf.int64)
        # Local assignments per node (shared head)
        logits = self.proj(Z)                          # [N_tot, K]
        S_local = self.act(logits)                     # [N_tot, K]

        # Build per-node graph ids: [0,0,...,0, 1,1,...,1, ...] length N_tot
        B = tf.shape(n_nodes)[0]
        gids = tf.repeat(tf.range(B, dtype = tf.int64), tf.cast(n_nodes, tf.int64))             # [N_tot]

        N_tot = tf.shape(Z, out_type = tf.int64)[0]

        # Row indices: each node row repeated K times
        rows = tf.tile(tf.range(N_tot, dtype = tf.int64)[:, None], [1, tf.cast(K, tf.int32)])   # [N_tot, K]

        # Column indices: for node i in graph g, cols = g*K + [0..K-1]
        cols_base = gids * K                                                                    # [N_tot]
        cols = cols_base[:, None] + tf.range(K, dtype = tf.int64)[None, :]                      # [N_tot, K]

        indices = tf.stack([tf.reshape(rows, [-1]), tf.reshape(cols, [-1])], axis = 1)          # [N_tot*K, 2]
        values = tf.reshape(S_local, [-1])                                                      # [N_tot*K]

        Sbd = tf.SparseTensor(
                                    indices = indices,
                                    values = tf.cast(values, tf.float32),
                                    dense_shape = [N_tot, tf.cast(B, tf.int64) * K]
                             )
        Sbd = tf.sparse.reorder(Sbd)
        Sbd_dense = tf.sparse.to_dense(Sbd)                                                     # [N_tot, K*B]

        # seg_ids for pooled graph (K rows per graph)
        seg_ids = tf.repeat(tf.range(B, dtype = tf.int32), repeats = self.K)                    # [K*B]
        return Sbd_dense, S_local, seg_ids

    def call(self, Z: tf.Tensor, A, n_nodes: tf.Tensor):
        """
        Z: [N_tot, F], A: [N_tot, N_tot] (dense or SparseTensor),
        n_nodes: [B]

        Returns:
            Zp: [K*B, F], Ap: [K*B, K*B], seg_ids: [K*B],
            S_local: [N_tot, K], Sbd_dense: [N_tot, K*B]
        """
        Sbd, S_local, seg_ids = self.build_S_blockdiag(Z, n_nodes)

        # X' = S^T Z
        Zp = tf.linalg.matmul(Sbd, Z, transpose_a = True)                                           # [K*B, F]

        # A' = S^T A S
        if isinstance(A, tf.SparseTensor):
            AS = tf.sparse.sparse_dense_matmul(A, Sbd)                                          # [N_tot, K*B]
            Ap = tf.linalg.matmul(Sbd, AS, transpose_a = True)                                  # [K*B, K*B]
        else:
            Ap = tf.linalg.matmul(tf.linalg.matmul(Sbd, A, transpose_a = True), Sbd)            # [K*B, K*B]

        return Zp, Ap, seg_ids, S_local, Sbd
