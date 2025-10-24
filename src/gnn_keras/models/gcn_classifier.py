from __future__ import annotations
import tensorflow as tf
from keras import Model, layers

try:
    from ..graph import Graph
    from ..layers import GraphConv
except Exception:
    from src.gnn_keras.graph import Graph
    from src.gnn_keras.layers import GraphConv

class GCNClassifier(Model):
    """Simple GCN for node classification (semi-supervised).

    Expects a single graph. Use `graph.mask` to select supervised nodes.
    """
    def __init__(self, hidden: int, classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GraphConv(hidden)
        self.do1 = layers.Dropout(dropout)
        self.conv2 = GraphConv(classes, activation = None, self_loops = True)

    def call(self, graph: Graph, training = None):
        g = self.conv1(graph, training = training)
        g = g.with_(x = self.do1(g.x, training = training))
        g = self.conv2(g, training = training)
        return g
