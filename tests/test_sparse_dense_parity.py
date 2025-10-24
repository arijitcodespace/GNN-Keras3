import tensorflow as tf
from gnn_keras import Graph
from gnn_keras.layers import GraphConv
from gnn_keras.ops import add_self_loops, normalize_adj

def test_conv_dense_sparse_parity():
    N, F, H = 16, 8, 4
    x = tf.random.normal([N, F])
    A = tf.random.uniform([N, N], 0, 2, dtype = tf.int32)
    A = tf.cast(A > 0, tf.float32)

    # sparse version
    idx = tf.where(A > 0)
    vals = tf.gather_nd(A, idx)
    As = tf.SparseTensor(indices = tf.cast(idx, tf.int64), values = vals, dense_shape = [N, N])

    g_dense = Graph(x = x, a = A)
    g_sparse = Graph(x = x, a = As)

    layer = GraphConv(H, activation = None)
    y_dense = layer(graph = g_dense).x
    y_sparse = layer(graph = g_sparse).x

    tf.debugging.assert_near(y_dense, y_sparse, atol = 7e-4)
