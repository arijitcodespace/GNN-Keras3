import tensorflow as tf
from gnn_keras import Graph
from gnn_keras.layers import GraphConv, GraphAttention

def test_gcn_output_shape():
    x = tf.random.normal([10, 5])
    A = tf.ones([10, 10])
    g = Graph(x = x, a = A)
    y = GraphConv(7, activation = None)(graph = g).x
    assert y.shape == (10, 7)

def test_gat_output_shape():
    x = tf.random.normal([10, 5])
    A = tf.ones([10, 10])
    g = Graph(x = x, a = A)
    y = GraphAttention(4, heads = 3, concat = True, activation = None)(graph = g).x
    assert y.shape == (10, 12)
