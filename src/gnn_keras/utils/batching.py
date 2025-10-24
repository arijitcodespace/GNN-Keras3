from __future__ import annotations
import tensorflow as tf
from typing import List

try:
    from ..graph import Graph
except Exception:
    from src.graph import Graph

def pack_block_diagonal(graphs: List[Graph]) -> Graph:
    """Pack small graphs into a block-diagonal big graph for batching.

    Concatenates features and builds block-diagonal adjacency (dense or sparse).
    """
    xs = [g.x for g in graphs]
    x = tf.concat(xs, axis = 0)
    # build sparse block-diagonal regardless of input type for generality
    rows = []
    cols = []
    vals = []
    n_offset = tf.constant(0, dtype = tf.int32)
    for g in graphs:
        if isinstance(g.a, tf.SparseTensor):
            idx = g.a.indices + tf.cast([n_offset, n_offset], tf.int64)
            rows.append(idx[:, 0])
            cols.append(idx[:, 1])
            vals.append(g.a.values)
            n = tf.cast(g.a.dense_shape[0], tf.int32)
        else:
            n = tf.shape(g.a)[0]
            nz = tf.where(g.a > 0)
            ii, jj = nz[:, 0], nz[:, 1]
            rows.append(tf.cast(ii + n_offset, tf.int64))
            cols.append(tf.cast(jj + n_offset, tf.int64))
            vals.append(tf.cast(tf.gather_nd(g.a, nz), tf.float32))
        n_offset = n_offset + n
    indices = tf.stack([tf.concat(rows, 0), tf.concat(cols, 0)], axis = 1)
    values = tf.concat(vals, 0)
    a = tf.SparseTensor(indices = indices, values = values, dense_shape = [tf.cast(n_offset, tf.int64), tf.cast(n_offset, tf.int64)])
    a = tf.sparse.reorder(a)
    return Graph(x = x, a = a)
