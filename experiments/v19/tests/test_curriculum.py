"""Regression tests for observed procedural-label failures."""
import numpy as np
import pytest
from hypernet_v19.curriculum import episode, program, digest


@pytest.mark.parametrize('split,index', [('dev', 7), ('compositional', 11), ('train', 99)])
def test_hidden_difference_is_resolved_without_replacing_target_type(split, index):
    row = episode(index, split)
    assert row['program_id'] == digest(program(index, split))
    assert min(row['metadata']['support_visible_edit_pixels']) >= 8
    assert min(row['metadata']['visible_edit_pixels']) >= 8


def test_absence_does_not_add_duplicate_program_cue():
    row = episode(1)
    ids = [item['program_id'] for item in row['metadata']['instances']]
    assert not row['positive']
    assert len(set(ids)) == len(ids)
    assert row['program_id'] not in ids
    assert not row['query_point_valid'].any()


def test_valid_correspondence_padding_does_not_repeat_locations():
    for index in (0, 2, 4):
        row = episode(index)
        points = row['support_points'][row['support_point_valid']]
        assert len(np.unique(points, axis=0)) == len(points)
        assert row['point_valid'].sum() >= 16
        assert not np.array_equal(row['support'], row['query'][:128, :128])
