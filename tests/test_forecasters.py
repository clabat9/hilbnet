"""Tests for HilbNetForecaster / STGNNConvForecaster.

The two classes must agree on output shape and on the regression head, so that
comparing them isolates the layer-stack choice (sheaf-Laplacian polynomial vs
spatio-temporal Conv1d).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilbnet.forecasters import (  # noqa: E402
    HilbNetForecaster,
    STGNNConvForecaster,
    masked_mae,
)


def _tiny_edge_index(n_nodes: int = 8) -> torch.Tensor:
    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))
    return torch.tensor([src, dst], dtype=torch.long)


class TestForecasterShapes(unittest.TestCase):
    def setUp(self):
        self.N, self.T_in, self.T_out, self.F_in = 8, 12, 12, 2
        self.B = 3
        self.edge_index = _tiny_edge_index(self.N)
        self.x = torch.randn(self.B, self.N, self.T_in, self.F_in)

    def _make_hilbnet(self, kappa=(2, 2), transport="circulant"):
        return HilbNetForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=list(kappa),
            transport_param_type=transport, num_householder_reflections=8,
        )

    def _make_stgnn_conv(self, kappa=(2, 2)):
        return STGNNConvForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=list(kappa), kernel_size=3,
        )

    def test_hilbnet_output_shape(self):
        m = self._make_hilbnet()
        out = m(self.x)
        self.assertEqual(tuple(out.shape), (self.B, self.N, self.T_out))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_stgnn_conv_output_shape(self):
        m = self._make_stgnn_conv()
        out = m(self.x)
        self.assertEqual(tuple(out.shape), (self.B, self.N, self.T_out))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_input_dim_check(self):
        m = self._make_hilbnet()
        with self.assertRaises(ValueError):
            m(torch.randn(self.B, self.N, self.T_in))  # missing F_in dim


class TestReadoutParityHilbNetVsSTGNNConv(unittest.TestCase):
    """The forecasters must differ ONLY in the layer stack — readout shape,
    head architecture, and forward contract are identical."""

    def setUp(self):
        torch.manual_seed(0)
        self.N, self.T_in, self.T_out, self.F_in = 8, 12, 12, 2
        self.B = 3
        self.edge_index = _tiny_edge_index(self.N)
        self.x = torch.randn(self.B, self.N, self.T_in, self.F_in)

    def test_output_shapes_agree(self):
        hn = HilbNetForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=[2, 2],
            transport_param_type="circulant",
        )
        st = STGNNConvForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=[2, 2],
        )
        self.assertEqual(hn(self.x).shape, st(self.x).shape)

    def test_head_param_shapes_agree(self):
        """The regression head should have the same shape across both classes."""
        hn = HilbNetForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=[2, 2],
            transport_param_type="circulant",
        )
        st = STGNNConvForecaster(
            n_nodes=self.N, time_steps_in=self.T_in, time_steps_out=self.T_out,
            edge_index=self.edge_index, in_features=[self.F_in, 4, 8],
            activation=nn.ReLU(), kappa=[2, 2],
        )
        # Head shapes must match exactly (same W_in, W_out per dim).
        hn_linear = [m for m in hn.head if isinstance(m, nn.Linear)][0]
        st_linear = [m for m in st.head if isinstance(m, nn.Linear)][0]
        self.assertEqual(hn_linear.in_features, st_linear.in_features)
        self.assertEqual(hn_linear.out_features, st_linear.out_features)


class TestKappaOneSkipsTransport(unittest.TestCase):
    """The kappa=[1,1] no-graph ablation (the MLP fiber baseline) skips
    transport allocation entirely, so its parameter count has no transports."""

    def test_no_transport_allocation(self):
        N, T_in, T_out = 6, 12, 12
        m = HilbNetForecaster(
            n_nodes=N, time_steps_in=T_in, time_steps_out=T_out,
            edge_index=_tiny_edge_index(N),
            in_features=[2, 4, 8], activation=nn.ReLU(), kappa=[1, 1],
        )
        self.assertFalse(m._uses_transport)
        self.assertIsNone(m.transport_maps)
        self.assertEqual(m.transport_parameters(), [])
        self.assertFalse(hasattr(m, "householder_vectors"))
        self.assertIsNone(m.transport_param_module)

        # Forward still works, returns finite outputs.
        x = torch.randn(2, N, T_in, 2)
        out = m(x)
        self.assertEqual(tuple(out.shape), (2, N, T_out))
        self.assertTrue(torch.isfinite(out).all().item())

        # kernel_penalty short-circuits to zero (no transport → no smooth-section).
        self.assertEqual(m.kernel_penalty(x).item(), 0.0)


class TestMaskedMAE(unittest.TestCase):
    def test_masked_mae_ignores_zero_mask(self):
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[2.0, 4.0, 6.0]])
        # mask out the second position (where the error is 2)
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        # |1-2| * 1 + |2-4| * 0 + |3-6| * 1 = 1 + 0 + 3 = 4. mask sum = 2.
        self.assertAlmostEqual(masked_mae(pred, target, mask).item(), 2.0, places=5)

    def test_all_zero_mask_returns_zero_not_nan(self):
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[5.0, 7.0]])
        mask = torch.tensor([[0.0, 0.0]])
        # Sum of masked errors is 0, mask sum clamped to 1 → 0/1 = 0.
        v = masked_mae(pred, target, mask).item()
        self.assertEqual(v, 0.0)
        # And not NaN/Inf.
        self.assertTrue(torch.isfinite(masked_mae(pred, target, mask)).item())


if __name__ == "__main__":
    unittest.main()
