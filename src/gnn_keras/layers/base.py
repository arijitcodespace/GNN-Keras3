from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers

try:
    from ..graph import Graph
except Exception:
    from src.gnn_keras.graph import Graph

class GraphLayer(layers.Layer):
    """Base layer that consumes/produces `Graph` objects.

    Subclasses should implement `_call_dense(x, a, training)` and
    `_call_sparse(x, a, training)` and return transformed node features.
    """
    
    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], Graph):
            graph = args[0]

            # If the user did layer(G, True) or layer(G, training=True),
            # fold the positional training into kwargs.
            if len(args) > 1 and 'training' not in kwargs:
                training_candidate = args[1]
                # Accept bool/None or a boolean Tensor
                is_bool_tensor = getattr(training_candidate, "dtype", None) is not None and str(training_candidate.dtype) == "bool"
                if isinstance(training_candidate, (bool, type(None))) or is_bool_tensor:
                    kwargs['training'] = training_candidate
                # Ignore other extra positionals (this layer doesn't expect them)

            # Call with keyword-only graph so Keras doesn't see a non-tensor positional
            return super().__call__(graph = graph, **kwargs)

        # Fallback: no special handling
        return super().__call__(*args, **kwargs)
    
    def call(self, *, graph: Graph = None, training = None):
        if graph is None:
            raise AttributeError("`graph` must be passed and cannot be None.")
        x, a = graph.x, graph.a
        if isinstance(a, tf.SparseTensor):
            out = self._call_sparse(x, a, training = training)
        else:
            out = self._call_dense(x, a, training = training)
        return graph.with_(x = out)

    # Default fallbacks to raise if not overridden
    def _call_dense(self, x, a, training = None):  # pragma: no cover - abstract
        raise NotImplementedError

    def _call_sparse(self, x, a, training = None):  # pragma: no cover - abstract
        raise NotImplementedError
