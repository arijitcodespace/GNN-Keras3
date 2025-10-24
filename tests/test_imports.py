def test_public_api():
    import gnn_keras as gk
    assert hasattr(gk, "Graph")
    assert hasattr(gk, "GraphConv")
    assert hasattr(gk, "GraphAttention")
    assert hasattr(gk, "DiffPool")
