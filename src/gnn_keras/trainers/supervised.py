from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

try:
    from ..graph import Graph
except Exception:
    from src.gnn_keras.graph import Graph

class NodeClassificationTrainer:
    """Minimal trainer for transductive node classification on a single graph."""
    def __init__(self, model, lr = 1e-2):
        self.model = model
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits = True)
        self.opt = optimizers.Adam(lr)
        self.acc = metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(self, graph: Graph):
        with tf.GradientTape() as tape:
            out = self.model(graph, training = True).x  # logits per node
            mask = graph.mask if graph.mask is not None else tf.ones(tf.shape(out)[0], dtype = tf.bool)
            loss = self.loss_fn(tf.boolean_mask(graph.y, mask), tf.boolean_mask(out, mask))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        self.acc.update_state(tf.boolean_mask(graph.y, mask), tf.boolean_mask(out, mask))
        return {"loss": loss, "acc": self.acc.result()}

    @tf.function
    def test_step(self, graph: Graph):
        out = self.model(graph, training = False).x
        mask = graph.mask if graph.mask is not None else tf.ones(tf.shape(out)[0], dtype = tf.bool)
        loss = self.loss_fn(tf.boolean_mask(graph.y, mask), tf.boolean_mask(out, mask))
        self.acc.update_state(tf.boolean_mask(graph.y, mask), tf.boolean_mask(out, mask))
        return {"loss": loss, "acc": self.acc.result()}
