"""Backend helpers.

Assumes Keras 3 with TensorFlow backend. You can extend this to support other
backends later if desired.
"""
from __future__ import annotations
import tensorflow as tf
try:
    import keras  # Keras 3
except Exception as e:  # pragma: no cover
    # Fallback to tf.keras if standalone keras isn't installed.
    from tensorflow import keras  # type: ignore

K = keras
TF = tf
