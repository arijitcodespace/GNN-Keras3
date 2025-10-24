from __future__ import annotations
import os, io, sys, tarfile, zipfile, urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from typing import List, Optional, Tuple

from gnn_keras import Graph

# ---------------------------
# TU dataset download helpers
# ---------------------------
TU_BASE = "https://www.chrsmrrs.com/graphkerneldatasets"


def _download_if_needed(root: Path, name: str) -> Path:
    root.mkdir(parents = True, exist_ok = True)
    ds_dir = root / name
    if ds_dir.exists() and any(ds_dir.glob("*.txt")):
        return ds_dir
    # Download zip
    url = f"{TU_BASE}/{name}.zip"
    zip_path = root / f"{name}.zip"
    print(f"Downloading {url} -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(ds_dir)
    return ds_dir


# -----------------
# File IO utilities
# -----------------

def _read_int_lines(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        vals = [int(x.strip()) for x in f]
    return np.asarray(vals, dtype = np.int64)


def _read_edge_list(path: Path) -> np.ndarray:
    # lines like "u, v" (1-indexed)
    edges = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, v = line.strip().split(",")
            edges.append((int(u), int(v)))
    return np.asarray(edges, dtype = np.int64)


def _read_node_attr(path: Path) -> np.ndarray:
    # comma‑separated floats per line
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = [float(x) for x in line.strip().split(",")]
            rows.append(parts)
    return np.asarray(rows, dtype = np.float32)


@dataclass
class TUGraph:
    x: np.ndarray  # [N, F]
    edge_index: np.ndarray  # [2, E] 0‑based local
    y: int


# -------------------------
# Parse TU format -> TUGraph
# -------------------------

def parse_tu_dataset(ds_dir: Path, name: str, one_hot_node_labels: bool = True,
                     concat_features: bool = True) -> Tuple[List[TUGraph], int]:
    """Reads TU raw files and returns per‑graph data.
    Returns (graphs, num_classes).
    """
    base = ds_dir / name
    edges = _read_edge_list(base.with_name(f"{name}_A.txt"))  # 1‑based global ids
    graph_indicator = _read_int_lines(base.with_name(f"{name}_graph_indicator.txt"))  # len = N
    graph_labels = _read_int_lines(base.with_name(f"{name}_graph_labels.txt"))       # len = G

    # Optional files
    node_labels_path = base.with_name(f"{name}_node_labels.txt")
    node_attrs_path  = base.with_name(f"{name}_node_attributes.txt")
    node_labels = _read_int_lines(node_labels_path) if node_labels_path.exists() else None
    node_attrs  = _read_node_attr(node_attrs_path) if node_attrs_path.exists() else None

    # Remap graph labels to 0..C-1
    uniq = np.unique(graph_labels)
    label_map = {v: i for i, v in enumerate(sorted(uniq))}
    y_all = np.vectorize(label_map.get)(graph_labels)

    # Prepare one‑hot for node labels if present
    x_label = None
    if node_labels is not None:
        nl_uniq = np.unique(node_labels)
        nl_map = {v: i for i, v in enumerate(sorted(nl_uniq))}
        node_labels_idx = np.vectorize(nl_map.get)(node_labels)
        K = len(nl_uniq)
        x_label = np.eye(K, dtype = np.float32)[node_labels_idx] if one_hot_node_labels else node_labels_idx[:, None].astype(np.float32)

    x_attr = node_attrs  # already float

    # Construct per‑graph
    G = int(y_all.shape[0])
    N = graph_indicator.shape[0]
    # Convert to 0‑based
    edges0 = edges - 1

    # Pre‑group nodes by graph
    nodes_by_g = [np.where(graph_indicator == (g+1))[0] for g in range(G)]  # 0‑based node ids
    node_to_local = [ {nid: i for i, nid in enumerate(nodes)} for nodes in nodes_by_g ]

    # Pre‑group edges by graph: keep edges with both ends in same graph
    graphs: List[TUGraph] = []

    for g in range(G):
        nodes = nodes_by_g[g]
        if nodes.size == 0:  # just in case
            continue
        node_set = set(nodes.tolist())
        # mask edges
        mask = np.array([(u in node_set) and (v in node_set) for (u, v) in edges0], dtype = bool)
        e = edges0[mask]
        # Relabel to local 0..n-1
        mapper = node_to_local[g]
        if e.size:
            src = np.vectorize(mapper.get)(e[:, 0])
            dst = np.vectorize(mapper.get)(e[:, 1])
            e_local = np.stack([src, dst], axis = 0)
            # Deduplicate undirected pairs
            und = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis = 1)
            und = np.unique(und, axis = 0)
            e_local = und.T  # [2, E]
        else:
            e_local = np.zeros((2, 0), dtype=np.int64)

        # Build node features
        feats = []
        if x_label is not None:
            feats.append(x_label[nodes])
        if x_attr is not None:
            feats.append(x_attr[nodes])
        if len(feats) == 0:
            # fallback: degree as a scalar feature
            deg = np.zeros((nodes.shape[0], 1), dtype = np.float32)
            for u, v in e_local.T:
                deg[u, 0] += 1
                deg[v, 0] += 1
            feats = [deg]
        x = np.concatenate(feats, axis=1) if (concat_features and len(feats) > 1) else feats[0]

        graphs.append(TUGraph(x = x.astype(np.float32), edge_index = e_local.astype(np.int64), y = int(y_all[g])))

    num_classes = len(uniq)
    return graphs, num_classes

def to_Graph(tg: TUGraph) -> Graph:
    # Build Graph from edge_index; add self‑loops/normalize in the conv layer
    x = tf.convert_to_tensor(tg.x, dtype = tf.float32)
    edge_index = tf.convert_to_tensor(tg.edge_index, dtype = tf.int64)
    G = Graph.from_edge_index(x, edge_index, num_nodes = x.shape[0], symmetric = True)
    return G.with_(y = tf.convert_to_tensor(tg.y, dtype = tf.int32))

def TUDataset(root, name):
    ds_dir = _download_if_needed(root, name)
    graphs, num_classes = parse_tu_dataset(ds_dir, name)
    return graphs, num_classes