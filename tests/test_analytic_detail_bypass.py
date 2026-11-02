"""Scientific controls for the separately frozen classical diagnostic."""
import importlib.util
from pathlib import Path
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'experiments/analytic_detail_bypass.py'
spec = importlib.util.spec_from_file_location('frozen_ncc_diagnostic', SOURCE)
ncc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncc)


def image():
    rgb = np.full((96, 96, 3), 30, np.uint8)
    rgb[18:78, 25:71] = 95
    rgb[32:37, 38:46] = 30
    rgb[57:65, 52:57] = 30
    return rgb


def test_one_support_bank_and_photometric_invariance():
    source = image(); box = [25, 18, 71, 78]
    bank = ncc.reference_bank(source, box)
    assert bank.shape == (72, 96*96)
    assert np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-6)
    # Positive affine brightness/contrast changes should cancel under NCC.
    query = source*2+10
    result = ncc.correlate(bank, query, [box])
    assert result['score'] > .99999
    assert result['bbox_xyxy'] == box


def test_flat_inputs_cannot_invent_a_pattern():
    flat = np.full((96, 96, 3), 123, np.uint8)
    box = [4.5, 5.2, 91.3, 89.7]
    bank = ncc.reference_bank(flat, box)
    assert not bank.any()
    assert ncc.correlate(bank, flat, [box])['score'] == 0


def test_no_proposals_returns_rejection_without_creating_boxes():
    source = image()
    bank = ncc.reference_bank(source, [25, 18, 71, 78])
    result = ncc.correlate(bank, source, [])
    assert result['score'] == -1 and result['bbox_xyxy'] is None
    assert result['proposals'] == result['proposal_scores'] == []


def test_preserves_predicted_proposal_coordinates_and_order():
    source = image(); boxes = [[0, 0, 13, 13], [25, 18, 71, 78]]
    bank = ncc.reference_bank(source, boxes[1])
    result = ncc.correlate(bank, source, boxes)
    assert result['proposals'] == boxes
    assert result['bbox_xyxy'] == boxes[1]
    assert result['proposal_scores'][1] > result['proposal_scores'][0]


def test_tied_calibration_never_exceeds_five_percent():
    for values in ([.2]*256, [.9]*13+[.1]*243, np.linspace(-1, 1, 256).tolist()):
        threshold = ncc.cutoff([{'score':score, 'present':False} for score in values])
        assert sum(value >= threshold for value in values) <= 12
