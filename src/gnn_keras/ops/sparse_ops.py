from __future__ import annotations
import tensorflow as tf
try:
    from ..typing import SparseTensor, Tensor
except Exception:
    from src.typing import SparseTensor, Tensor

def edge_index_from_sparse(a: SparseTensor) -> Tensor:
    """Return edge_index [2, E] from a tf.SparseTensor adjacency."""
    idx = tf.transpose(tf.cast(a.indices, tf.int32))  # [2,E]
    return idx

def coalesce_sum(indices: Tensor, values: Tensor, num_nodes: int) -> SparseTensor:
    """Coalesce duplicate edges by summing values; returns SparseTensor."""
    st = tf.SparseTensor(indices = tf.cast(tf.transpose(indices), tf.int64),
                         values = values, dense_shape = [num_nodes, num_nodes])
    return tf.sparse.reorder(st)
