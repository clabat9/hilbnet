import torch

from hilbnet.utils import (
    apply_sheaf_laplacian_from_transport,
    build_householder_transport_maps,
    build_sheaf_laplacian_from_transport,
    canonicalize_edge_index,
)


def test_canonicalize_edge_index_removes_duplicates_and_orients_edges():
    edge_index = torch.tensor([[1, 0, 2, 2, 1], [0, 1, 1, 1, 2]])
    oriented = canonicalize_edge_index(edge_index)
    expected = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(oriented.edge_index, expected)


def test_sheaf_laplacian_is_symmetric_psd_and_has_zero_non_edges():
    edge_index = torch.tensor([[0, 1], [1, 2]])
    P = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
        ]
    )
    L = build_sheaf_laplacian_from_transport(P, edge_index, stalk_dim=2, num_nodes=4)
    assert torch.allclose(L, L.T, atol=1e-6)
    assert torch.linalg.eigvalsh(L).min().item() >= -1e-6  # positive semidefinite

    # Node 0 and node 3 are not adjacent, so their block must be zero.
    block_03 = L.reshape(4, 2, 4, 2).permute(0, 2, 1, 3)[0, 3]
    assert torch.allclose(block_03, torch.zeros_like(block_03))


def test_sparse_apply_matches_dense_for_identity_transport():
    B, N, T, F = 1, 3, 2, 1
    x = torch.arange(B * N * T * F, dtype=torch.float32).reshape(B, N, T, F)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    P = torch.eye(T).unsqueeze(0).repeat(edge_index.shape[1], 1, 1)
    dense = build_sheaf_laplacian_from_transport(P, edge_index, stalk_dim=T, num_nodes=N)
    dense_out = torch.matmul(dense.unsqueeze(0), x.reshape(B, N * T, F)).reshape(B, N, T, F)
    sparse_out = apply_sheaf_laplacian_from_transport(x, P, edge_index)
    assert torch.allclose(dense_out, sparse_out, atol=1e-6)


def test_householder_transport_maps_are_orthogonal():
    torch.manual_seed(0)
    householder_vectors = torch.randn(3, 4, 5)
    transport_maps = build_householder_transport_maps(householder_vectors)
    gram = transport_maps.transpose(-1, -2) @ transport_maps
    eye = torch.eye(5).unsqueeze(0).repeat(3, 1, 1)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_orthogonal_fast_path_matches_general_transport_path():
    torch.manual_seed(0)
    B, N, T, F = 2, 4, 3, 2
    x = torch.randn(B, N, T, F)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]])
    edge_weights = torch.tensor([1.0, 0.7, 1.2])
    householder_vectors = torch.randn(edge_index.shape[1], 3, T)
    transport_maps = build_householder_transport_maps(householder_vectors)

    general_out = apply_sheaf_laplacian_from_transport(
        x,
        transport_maps=transport_maps,
        edge_index=edge_index,
        edge_weights=edge_weights,
        assume_orthogonal=False,
    )
    orthogonal_out = apply_sheaf_laplacian_from_transport(
        x,
        transport_maps=transport_maps,
        edge_index=edge_index,
        edge_weights=edge_weights,
        assume_orthogonal=True,
    )
    assert torch.allclose(general_out, orthogonal_out, atol=1e-5)

    dense_general = build_sheaf_laplacian_from_transport(
        transport_maps,
        edge_index,
        stalk_dim=T,
        edge_weights=edge_weights,
        num_nodes=N,
        assume_orthogonal=False,
    )
    dense_orthogonal = build_sheaf_laplacian_from_transport(
        transport_maps,
        edge_index,
        stalk_dim=T,
        edge_weights=edge_weights,
        num_nodes=N,
        assume_orthogonal=True,
    )
    assert torch.allclose(dense_general, dense_orthogonal, atol=1e-5)
