from __future__ import annotations
import tensorflow as tf
try:
    from ..typing import Adjacency, Tensor, SparseTensor
except Exception:
    from src.typing import Adjacency, Tensor, SparseTensor

def _eye(n: int, dtype) -> Tensor:
    return tf.eye(n, dtype = dtype)

def add_self_loops(a: Adjacency, fill: float = 1.0) -> Adjacency:
    """Add self-loops to dense or sparse adjacency."""
    if isinstance(a, tf.SparseTensor):
        n = a.dense_shape[0]
        rng = tf.range(n, dtype = tf.int64)
        idx = tf.stack([rng, rng], axis=1)
        values = tf.fill([tf.cast(n, tf.int32)], tf.cast(fill, a.dtype))
        eye = tf.sparse.SparseTensor(indices = idx, values = values, dense_shape = a.dense_shape)
        eye = tf.sparse.reorder(eye)
        return tf.sparse.add(a, eye)
    else:
        n = tf.shape(a)[0]
        return a + _eye(tf.cast(n, tf.int32), a.dtype) * tf.cast(fill, a.dtype)

@tf.function
def normalize_adj(a: Adjacency, symmetric: bool = True, eps: float = 1e-9) -> Adjacency:
    """D^{-1/2} A D^{-1/2} (symmetric) or D^{-1} A (random-walk).

    Works for dense and tf.SparseTensor.
    """
    if isinstance(a, tf.SparseTensor):
        # degree = sum of rows
        deg = tf.sparse.reduce_sum(a, axis = 1)  # [N]
        if symmetric:
            inv_sqrt = tf.math.rsqrt(tf.maximum(deg, eps))
            D_left = tf.gather(inv_sqrt, a.indices[:, 0])
            D_right = tf.gather(inv_sqrt, a.indices[:, 1])
            values = a.values * tf.cast(D_left, a.dtype) * tf.cast(D_right, a.dtype)
        else:
            inv = 1.0 / tf.maximum(deg, eps)
            D_left = tf.gather(inv, a.indices[:, 0])
            values = a.values * tf.cast(D_left, a.dtype)
        return tf.sparse.reorder(tf.SparseTensor(indices = a.indices, values = values, dense_shape = a.dense_shape))
    else:
        deg = tf.reduce_sum(a, axis = 1)
        if symmetric:
            inv_sqrt = tf.math.rsqrt(tf.maximum(deg, eps))
            D = tf.linalg.diag(inv_sqrt)
            return D @ a @ D
        else:
            inv = 1.0 / tf.maximum(deg, eps)
            D = tf.linalg.diag(inv)
            return D @ a
