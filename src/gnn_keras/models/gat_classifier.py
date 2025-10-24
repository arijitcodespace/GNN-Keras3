from __future__ import annotations
import tensorflow as tf
from keras import Model, layers

try:
    from ..graph import Graph
    from ..layers import GraphAttention
except Exception:
    from src.gnn_keras.graph import Graph
    from src.gnn_keras.layers import GraphAttention

class GATClassifier(Model):
    def __init__(self, hidden: int, classes: int, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.gat1 = GraphAttention(hidden, heads = heads, concat = True)
        self.do1 = layers.Dropout(dropout)
        self.gat2 = GraphAttention(classes, heads = 1, concat = False, activation = None)

    def call(self, graph: Graph, training=None):
        g = self.gat1(graph, training = training)
        g = g.with_(x = self.do1(g.x, training = training))
        g = self.gat2(g, training = training)
        return g
