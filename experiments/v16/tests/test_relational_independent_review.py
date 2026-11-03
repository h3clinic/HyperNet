"""Independent checks of learning paths and coordinates, not benchmark tests."""
import numpy as np
import torch
from torch.nn import functional as F
from hypernet.relational_encoder import ChannelNorm, RelationalEncoder, objective, sample_nodes, supervised_contrast
from hypernet.variation_curriculum import EDGES, catalog, render


def small_batch():
    torch.set_num_threads(2)
    codes = np.repeat(catalog()[:2], 2)
    return render(codes, np.array([31, 53, 71, 97], np.uint64), size=64)


def test_relational_objective_updates_actual_dense_descriptor_and_shared_stem():
    torch.manual_seed(16); batch = small_batch(); model = RelationalEncoder()
    out = model(batch['rgb'].permute(0, 3, 1, 2).float() / 255)
    degrees, edges = model.relations(out, batch['node_xy'], torch.as_tensor(EDGES))
    loss = F.cross_entropy(degrees.flatten(0, 1), batch['degrees'].flatten()) + F.binary_cross_entropy_with_logits(edges, batch['edges'])
    loss.backward()
    assert model.dense_head.weight.grad.abs().sum() > 0
    assert model.stem[0].weight.grad.abs().sum() > 0
    assert model.embedding_head[0].weight.grad is None  # No detached descriptor surrogate.


def test_type_invariance_loss_uses_same_dense_descriptor_and_mask_pooling():
    torch.manual_seed(17); batch = small_batch(); model = RelationalEncoder()
    out = model(batch['rgb'].permute(0, 3, 1, 2).float() / 255)
    loss, _ = supervised_contrast(out['embedding'], batch['type_codes']); loss.backward()
    assert model.dense_head.weight.grad.abs().sum() > 0
    assert model.mask_head.weight.grad.abs().sum() > 0
    assert model.stem[0].weight.grad.abs().sum() > 0


def test_relation_ablation_changes_only_declared_loss_terms_on_same_outputs():
    torch.manual_seed(18); batch = small_batch(); model = RelationalEncoder()
    out = model(batch['rgb'].permute(0, 3, 1, 2).float() / 255)
    full, stats = objective(model, out, batch, torch.as_tensor(EDGES), relations=True)
    ablated, other = objective(model, out, batch, torch.as_tensor(EDGES), relations=False)
    expected = .35 * stats['adjacency_loss'] + .15 * stats['degree_loss']
    assert torch.isclose(full - ablated, torch.tensor(expected), rtol=1e-5, atol=1e-6)
    for key in stats:
        if key != 'loss': assert stats[key] == other[key]


def test_node_gather_matches_grid_sample_values_and_gradients():
    torch.manual_seed(19); x = torch.randn(2, 4, 7, 9, requires_grad=True)
    grid = torch.rand(2, 16, 2) * 2.2 - 1.1
    actual = sample_nodes(x, grid)
    expected = F.grid_sample(x, grid[:, None], align_corners=False, padding_mode='zeros').squeeze(2).transpose(1, 2)
    assert torch.allclose(actual, expected, atol=1e-6)
    ga, = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)
    ge, = torch.autograd.grad(expected.square().sum(), x)
    assert torch.allclose(ga, ge, atol=2e-6)


def test_channel_normalization_has_no_remote_spatial_statistics():
    torch.manual_seed(20); layer = ChannelNorm(8); x = torch.randn(2, 8, 5, 6)
    extended = torch.cat([x, torch.randn(2, 8, 5, 7) * 100], dim=3).requires_grad_()
    # Width-dependent kernel reduction order can change last bits; remote
    # dependence is tested exactly through zero cross-location derivatives.
    assert torch.allclose(layer(x), layer(extended)[:, :, :, :6], atol=1e-6, rtol=1e-6)
    gradient, = torch.autograd.grad(layer(extended)[:, :, :, :6].square().sum(), extended)
    assert torch.count_nonzero(gradient[:, :, :, 6:]) == 0
