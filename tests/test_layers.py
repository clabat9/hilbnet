import torch

from hilbnet.layers import HilbertConvLayer
from hilbnet.utils import apply_sheaf_laplacian_from_transport, build_sheaf_laplacian_from_transport


def test_edgewise_laplacian_action_matches_dense_operator():
    torch.manual_seed(0)
    B, N, T, F = 2, 4, 3, 2
    x = torch.randn(B, N, T, F)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    edge_weights = torch.tensor([1.0, 0.7, 1.2, 0.5])
    transport_maps = torch.randn(edge_index.shape[1], T, T)

    dense_L = build_sheaf_laplacian_from_transport(
        transport_maps,
        edge_index,
        stalk_dim=T,
        edge_weights=edge_weights,
        num_nodes=N,
        return_blocks=False,
    )
    x_flat = x.reshape(B, N * T, F)
    dense_out = torch.matmul(dense_L.unsqueeze(0), x_flat).reshape(B, N, T, F)

    sparse_out = apply_sheaf_laplacian_from_transport(
        x,
        transport_maps=transport_maps,
        edge_index=edge_index,
        edge_weights=edge_weights,
    )
    assert torch.allclose(sparse_out, dense_out, atol=1e-6)


def test_layer_matches_polynomial_filter_defined_by_edgewise_laplacian_for_kappa_2():
    torch.manual_seed(0)
    B, N, T, Fin, Fout = 2, 3, 2, 1, 2
    x = torch.randn(B, N, T, Fin)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    edge_weights = torch.tensor([1.0, 0.5, 1.5])
    transport_maps = torch.randn(edge_index.shape[1], T, T)
    layer = HilbertConvLayer(in_channels=Fin, out_channels=Fout, kappa=2, activation=torch.nn.Identity())
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[2.0, -1.0]], [[0.5, 1.5]]]))
        layer.bias.zero_()

    y = layer(x, transport_maps=transport_maps, edge_index=edge_index, edge_weights=edge_weights)

    expected0 = torch.einsum('bntf,fg->bntg', x, layer.weight[0])
    Lx = apply_sheaf_laplacian_from_transport(x, transport_maps=transport_maps, edge_index=edge_index, edge_weights=edge_weights)
    expected1 = torch.einsum('bntf,fg->bntg', Lx, layer.weight[1])
    expected = expected0 + expected1
    assert torch.allclose(y, expected, atol=1e-6)
