from __future__ import annotations
import tensorflow as tf
from ..graph import Graph
from ..ops import normalize_adj, add_self_loops

class ToyCitation:
    """Small in-memory graph for examples/tests (no SciPy).

    Generates a random citation-like sparse graph.
    """
    def __init__(self, n_nodes=2708, n_feats=1433, n_edges=10556, n_classes=7):
        self.n = n_nodes
        self.f = n_feats
        self.c = n_classes
        self.e = n_edges

    def graph(self) -> Graph:
        x = tf.random.normal([self.n, self.f])
        src = tf.random.uniform([self.e], 0, self.n, dtype=tf.int32)
        dst = tf.random.uniform([self.e], 0, self.n, dtype=tf.int32)
        edge_index = tf.stack([src, dst], axis=0)
        G = Graph.from_edge_index(x, edge_index, num_nodes=self.n)
        a = normalize_adj(add_self_loops(G.a))
        y = tf.random.uniform([self.n], 0, self.c, dtype=tf.int32)
        return G.with_(a=a, y=y)
