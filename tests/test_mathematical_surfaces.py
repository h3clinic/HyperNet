"""Geometric rendering, independent RNG streams, and artifact regression tests."""
from collections import deque
import numpy as np
import pytest

from hypernet.mathematical_surfaces import (
    FAMILY_NAMES, FAMILY_TO_INDEX, render_surface, _rasterize_triangles,
    _superangle, _spow,
)


@pytest.mark.parametrize('family', FAMILY_NAMES)
def test_all_named_families_have_native_finite_geometric_outputs(family):
    result = render_surface(family, 109, 5, .07, 64)
    assert result['rgb'].shape == (64, 64, 3) and result['rgb'].dtype == np.uint8
    assert result['mask'].shape == (64, 64) and result['mask'].dtype == np.bool_
    assert .02 < result['mask'].mean() < .75
    assert result['normal'].shape == (64, 64, 3) and result['normal'].dtype == np.float32
    assert result['depth'].shape == (64, 64) and result['depth'].dtype == np.float32
    assert np.isfinite(result['normal']).all() and np.isfinite(result['depth']).all()
    mask = result['mask']
    np.testing.assert_allclose(np.linalg.norm(result['normal'][mask], axis=1), 1., atol=2e-6)
    assert np.all(result['normal'][mask, 2] >= 0)
    assert (result['depth'][mask] > 0).all()
    assert (result['depth'][~mask] == 0).all() and (result['normal'][~mask] == 0).all()
    assert (result['component'][~mask] == 255).all()
    assert result['family_index'] == FAMILY_TO_INDEX[family]
    assert result['render']['native_resolution'] == 64
    assert result['render']['image_resizing'] is False and result['render']['gaussian_image_blur'] is False
    assert result['render']['external_assets'] == result['render']['natural_images'] == 0
    assert result['render']['surface_triangles'] > 0


@pytest.mark.parametrize('family', FAMILY_NAMES)
def test_surface_treatment_is_bounded_geometry_not_image_noise(family):
    smooth = render_surface(family, 44, 12, 0., 64)
    detail = render_surface(family, 44, 12, .12, 64)
    assert smooth['base_program_id'] == detail['base_program_id']
    assert smooth['program_id'] != detail['program_id']
    assert smooth['geometry_sha256'] != detail['geometry_sha256']
    assert smooth['view'] == detail['view']
    assert smooth['parameters']['gaussian_bumps'] == detail['parameters']['gaussian_bumps']
    assert smooth['parameters']['maximum_actual_displacement'] == 0.
    assert 0 < detail['parameters']['maximum_actual_displacement'] <= .12+1e-6
    assert np.any(smooth['depth'] != detail['depth'])
    assert np.any(smooth['normal'] != detail['normal'])
    assert np.any(smooth['rgb'] != detail['rgb'])


def test_geometry_view_streams_are_independent_and_reproducible():
    first = render_surface('trefoil_tube', 13, 17, .07, 64)
    same = render_surface('trefoil_tube', 13, 17, .07, 64)
    moved = render_surface('trefoil_tube', 13, 18, .07, 64)
    new_object = render_surface('trefoil_tube', 14, 17, .07, 64)
    for key in ('rgb', 'mask', 'normal', 'depth', 'component'):
        np.testing.assert_array_equal(first[key], same[key])
    assert first['program_id'] == moved['program_id']
    assert first['geometry_sha256'] == moved['geometry_sha256']
    assert first['parameters'] == moved['parameters']
    assert first['view'] != moved['view'] and np.any(first['rgb'] != moved['rgb'])
    assert first['view'] == new_object['view']
    assert first['base_program_id'] != new_object['base_program_id']
    assert first['geometry_sha256'] != new_object['geometry_sha256']


def test_program_identity_is_resolution_independent_and_output_is_native():
    low = render_surface('supertoroid', 87, 3, .06, 64)
    high = render_surface('supertoroid', 87, 3, .06, 128)
    assert low['program_id'] == high['program_id']
    assert low['base_program_id'] == high['base_program_id']
    assert low['geometry_sha256'] != high['geometry_sha256']
    assert high['rgb'].shape == (128, 128, 3)
    assert not np.array_equal(high['rgb'], np.repeat(np.repeat(low['rgb'], 2, 0), 2, 1))


def test_triangle_coverage_has_no_holes_or_one_pixel_splat_growth():
    # Two triangles exactly fill [1,3]x[1,3], so only four pixel centers are inside.
    points = np.array([[1, 1, .2], [3, 1, .2], [3, 3, .2], [1, 3, .2]], dtype=np.float32)
    normals = np.tile([0., 0., 1.], (4, 1)).astype(np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mask, depth, normal, parts = _rasterize_triangles(points, normals, triangles,
        np.zeros(4, dtype=np.uint8), points[:, 0], points[:, 1], 5)
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(mask.reshape(5, 5), expected)
    np.testing.assert_allclose(depth, .2)
    np.testing.assert_allclose(normal, np.tile([0., 0., 1.], (4, 1)))
    assert np.all(parts == 0)


def test_depth_buffer_selects_front_surface_with_interpolated_normals():
    back = np.array([[0, 0, .1], [3, 0, .1], [0, 3, .1]], dtype=np.float32)
    front = back.copy()
    front[:, 2] = .5
    points = np.concatenate((back, front))
    normals = np.tile([0., 0., 1.], (6, 1)).astype(np.float32)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    parts = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    mask, depths, _, selected_parts = _rasterize_triangles(points, normals, triangles,
        parts, points[:, 0], points[:, 1], 4)
    assert mask.any() and np.all(selected_parts == 1)
    np.testing.assert_allclose(depths, .5)


def test_inverse_power_sampling_removes_cusp_spacing_regression():
    angle = np.linspace(0, np.pi/2, 65, dtype=np.float32)
    uniform = _spow(np.sin(angle), .55)
    adaptive = _spow(np.sin(_superangle(angle, .55)), .55)
    assert np.diff(adaptive).max() < .5*np.diff(uniform).max()
    assert np.all(np.diff(adaptive) >= -1e-6)


def test_closed_smooth_superellipsoid_has_no_internal_raster_holes():
    for seed in (410, 9, 93):
        mask = render_surface('superellipsoid', seed, 18, 0., 64)['mask']
        outside = np.zeros_like(mask)
        queue = deque([(0, 0)])
        while queue:
            y, x = queue.popleft()
            if not (0 <= y < 64 and 0 <= x < 64) or mask[y, x] or outside[y, x]:
                continue
            outside[y, x] = True
            queue.extend(((y-1, x), (y+1, x), (y, x-1), (y, x+1)))
        assert np.all(mask | outside)


@pytest.mark.parametrize('kwargs', [
    {'family': 'square'}, {'object_seed': -1}, {'object_seed': True}, {'view_seed': 1.2},
    {'size': 12}, {'size': 400}, {'size': True}, {'surface_strength': float('nan')},
    {'surface_strength': -.01}, {'surface_strength': .3}, {'surface_strength': True},
])
def test_invalid_parameters_rejected(kwargs):
    values = {'family': 'superellipsoid', 'object_seed': 1, 'size': 64, **kwargs}
    with pytest.raises(ValueError):
        render_surface(**values)


def test_returned_parameter_mutation_does_not_change_cached_geometry():
    result = render_surface('saddle_shell', 11, 5, .07, 64)
    original = result['parameters']['gaussian_bumps'][0]['amplitude']
    result['parameters']['gaussian_bumps'][0]['amplitude'] = 999.
    again = render_surface('saddle_shell', 11, 5, .07, 64)
    assert again['parameters']['gaussian_bumps'][0]['amplitude'] == original


@pytest.mark.parametrize('family', FAMILY_NAMES)
def test_object_seed_changes_base_geometry_even_without_surface_treatment(family):
    first = render_surface(family, 110, 5, 0., 64)
    second = render_surface(family, 111, 5, 0., 64)
    assert first['view'] == second['view']
    assert first['geometry_sha256'] != second['geometry_sha256']
    assert np.any(first['depth'] != second['depth'])
