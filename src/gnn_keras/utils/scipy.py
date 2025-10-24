from __future__ import annotations
from typing import Optional
import numpy as np
import tensorflow as tf

def _require_scipy():
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
                            "scipy is required for these helpers. Install with `pip install scipy`."
                         ) from e
    return sp

def coo_to_tf_sparse(coo_matrix, dtype: Optional[tf.dtypes.DType] = None) -> tf.SparseTensor:
    """Convert a SciPy COO matrix to a `tf.SparseTensor` (indices int64)."""
    sp = _require_scipy()
    if not sp.isspmatrix_coo(coo_matrix):
        raise TypeError("Expected scipy.sparse.coo_matrix")
    indices = np.stack([coo_matrix.row, coo_matrix.col], axis = 1).astype(np.int64, copy = False)
    values = coo_matrix.data
    tf_values = tf.cast(values, dtype = dtype) if dtype is not None else tf.convert_to_tensor(values)
    st = tf.SparseTensor(indices = indices, values = tf_values, dense_shape = coo_matrix.shape)
    return tf.sparse.reorder(st)

def csr_to_tf_sparse(csr_matrix, dtype: Optional[tf.dtypes.DType] = None) -> tf.SparseTensor:
    """Convert a SciPy CSR matrix to a `tf.SparseTensor` via COO view."""
    sp = _require_scipy()
    if not sp.isspmatrix_csr(csr_matrix):
        raise TypeError("Expected scipy.sparse.csr_matrix")
    return coo_to_tf_sparse(csr_matrix.tocoo(copy = False), dtype = dtype)

def any_scipy_to_tf_sparse(sparse_matrix, dtype: Optional[tf.dtypes.DType] = None) -> tf.SparseTensor:
    """Convert any SciPy sparse (CSR/CSC/COO/...) to `tf.SparseTensor`."""
    sp = _require_scipy()
    if sp.isspmatrix_coo(sparse_matrix):
        return coo_to_tf_sparse(sparse_matrix, dtype = dtype)
    return coo_to_tf_sparse(sparse_matrix.tocoo(copy = False), dtype = dtype)

def scipy_edge_index(sparse_matrix) -> tf.Tensor:
    """Return edge_index [2, E] (int32) from a SciPy sparse matrix."""
    sp = _require_scipy()
    coo = sparse_matrix if sp.isspmatrix_coo(sparse_matrix) else sparse_matrix.tocoo(copy = False)
    idx = np.stack([coo.row.astype(np.int32, copy = False), coo.col.astype(np.int32, copy = False)], axis = 0)
    return tf.convert_to_tensor(idx, dtype = tf.int32)
