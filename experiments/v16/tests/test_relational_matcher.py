import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypernet.relational_matcher import score_vectors, validate_rgb


def test_two_scale_recipe_has_no_hidden_selection_or_raw_fusion():
    refs = {'native': np.eye(2, dtype=np.float32), 'context128': np.eye(2, dtype=np.float32)}
    query = {'native': np.array([1., 0.], np.float32), 'context128': np.array([0., 1.], np.float32)}
    scores = score_vectors(refs, query)
    np.testing.assert_array_equal(scores['native'], [1, 0])
    np.testing.assert_array_equal(scores['context128'], [0, 1])
    np.testing.assert_array_equal(scores['mean'], [.5, .5])


def test_invalid_pixels_and_nonfinite_embeddings_fail_closed():
    with pytest.raises(ValueError): validate_rgb(np.zeros((32, 32, 3), np.float32))
    with pytest.raises(ValueError): validate_rgb(np.zeros((8, 32, 3), np.uint8))
    with pytest.raises(ValueError):
        score_vectors({'native': np.array([[np.nan, 0]]), 'context128': np.ones((1, 2))},
                      {'native': np.ones(2), 'context128': np.ones(2)})


def test_other_query_images_do_not_change_the_current_image_embedding():
    import torch
    from hypernet.relational_encoder import RelationalEncoder
    from hypernet.relational_matcher import FrozenRelations
    torch.set_num_threads(2); torch.manual_seed(1615)
    extractor = FrozenRelations.__new__(FrozenRelations)
    extractor.device = 'cpu'; extractor.model = RelationalEncoder().eval().requires_grad_(False)
    rng = np.random.default_rng(16)
    a, b, c = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    single = extractor.encode(a)
    paired = extractor.encode_many([a, b, c])[0]
    for key in ('native', 'context128'):
        np.testing.assert_allclose(single[key], paired[key], atol=1e-6, rtol=1e-6)
    assert single['rgb_sha256'] == paired['rgb_sha256']
