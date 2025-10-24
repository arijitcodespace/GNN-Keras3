# gnn-keras3

Graph Neural Networks (GCN, GAT, DiffPool) built on TensorFlow + Keras 3 with efficient sparse ops via `tf.sparse`. No SciPy dependency.

## Features
- Unified `Graph` container supporting dense and `tf.SparseTensor` adjacencies
- Layers: GraphConv (GCN), GraphAttention (GAT), DiffPool (assignment + coarsening)
- Ops: degree normalization, self-loops, batching, edge-index → `tf.SparseTensor`
- Import-friendly API: `from gnn_keras import Graph, GraphConv, GraphAttention, DiffPool, GCNClassifier`
- Examples and tests

## Quick start
```bash
pip install -e .[dev]
```

```python
import tensorflow as tf
from gnn_keras import Graph, GraphConv, ops

# toy dense graph
x = tf.random.normal([5, 16])
A = ops.add_self_loops(tf.ones([5,5]))
G = Graph(x=x, a=A)

conv = GraphConv(32)
y = conv(G)  # [5,32]
```

Set Keras backend (if needed):
```python
import keras
keras.config.set_backend("tensorflow")
```
