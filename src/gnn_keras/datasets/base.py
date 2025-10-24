from __future__ import annotations
from typing import Iterator, Optional
import tensorflow as tf
from ..graph import Graph

class GraphDataset:
    """Minimal dataset protocol returning Graph objects.

    Implement `__iter__` for graph-level tasks or provide a single Graph via
    `.graph()` for transductive node tasks.
    """
    def __iter__(self) -> Iterator[Graph]:  # graph-level batches
        raise NotImplementedError

    def graph(self) -> Graph:  # single-graph tasks
        raise NotImplementedError
