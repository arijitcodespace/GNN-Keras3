from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, activations

try:
    from .base import GraphLayer
    from ..ops import normalize_adj, add_self_loops
except Exception:
    from src.gnn_keras.layers.base import GraphLayer
    from src.gnn_keras.ops import normalize_adj, add_self_loops
    
class GraphConv(GraphLayer):
    """GCN layer (Kipf & Welling style).

    Supports dense `tf.Tensor` and `tf.SparseTensor` adjacencies.
    """
    def __init__(self, units: int, use_bias: bool = True, activation = "relu", normalize = True, self_loops = True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.normalize = normalize
        self.self_loops = self_loops
        self.lin = layers.Dense(units, use_bias = use_bias)

    def _maybe_prep(self, a):
        if self.self_loops:
            a = add_self_loops(a)
        if self.normalize:
            a = normalize_adj(a)
        return a

    def _call_dense(self, x, a, training = None):
        a = self._maybe_prep(a)
        xw = self.lin(x)
        return tf.linalg.matmul(a, xw)

    def _call_sparse(self, x, a, training = None):
        a = self._maybe_prep(a)
        xw = self.lin(x)
        return tf.sparse.sparse_dense_matmul(a, xw)
