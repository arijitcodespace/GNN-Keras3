from __future__ import annotations
from typing import Optional, Union
import tensorflow as tf
from tensorflow.experimental import ExtensionType

try:
    from .typing import Tensor, SparseTensor, Adjacency
except Exception:
    from src.gnn_keras.typing import Tensor, SparseTensor, Adjacency

class Graph(ExtensionType):
    """Immutable graph container.

    Attributes
    -----------
    x : tf.Tensor [N, F]
        Node features.
    a : Union[tf.Tensor [N,N], tf.SparseTensor]
        Adjacency (dense or sparse). Self-loops optional.
    y : Optional[tf.Tensor]
        Labels for nodes or graph.
    mask : Optional[tf.Tensor]
        Boolean mask for nodes when doing semi-supervised tasks.
    """
    x: Tensor
    a: Adjacency
    y: Optional[tf.Tensor] = None
    mask: Optional[tf.Tensor] = None

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.a, tf.SparseTensor)

    def with_(self, **kwargs) -> "Graph":
        fields = dict(x = self.x, a = self.a, y = self.y, mask = self.mask)
        fields.update(kwargs)
        return Graph(**fields)

    @staticmethod
    def from_edge_index(
                            x: Tensor,
                            edge_index: tf.Tensor,  # [2, E] integer indices
                            num_nodes: Optional[int] = None,
                            edge_weight: Optional[Tensor] = None,
                            symmetric: bool = True,
                        ) -> "Graph":
        
        """Build a Graph from a COO-like edge_index using tf.sparse only."""
        if num_nodes is None:
            num_nodes = tf.reduce_max(edge_index) + 1
        indices = tf.transpose(edge_index)  # [E,2]
        values = tf.ones(tf.shape(indices)[0], dtype = x.dtype) if edge_weight is None else edge_weight
        a = tf.SparseTensor(indices = tf.cast(indices, tf.int64),
                            values = tf.cast(values, x.dtype),
                            dense_shape = [num_nodes, num_nodes])
        a = tf.sparse.reorder(a)
        if symmetric:
            a_t = tf.sparse.transpose(a)
            a = tf.sparse.add(a, a_t)
        return Graph(x = x, a = a)
