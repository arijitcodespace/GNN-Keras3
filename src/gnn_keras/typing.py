from __future__ import annotations
from typing import Optional, Union, Tuple
import tensorflow as tf

Tensor = tf.Tensor
SparseTensor = tf.SparseTensor
Adjacency = Union[Tensor, SparseTensor]
Shape2D = Tuple[int, int]

__all__ = ["Tensor", "SparseTensor", "Adjacency", "Shape2D"]
