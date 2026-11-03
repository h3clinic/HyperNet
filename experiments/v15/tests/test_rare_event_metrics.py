import numpy as np
import pytest
from hypernet.rare_event_metrics import calibration_cutoff,summary,upper_bound,projected_search

def test_zero_errors_does_not_certify_zero_risk():
    assert upper_bound(0,6000)==pytest.approx(1-.05**(1/6000))
    assert 4.9<10000*upper_bound(0,6000)<5.1

def test_tied_cutoff_is_conservative():
    result=calibration_cutoff([.8,.8,.5,.4],2500)
    assert result['observed_false_alarms']==0
    assert result['threshold']>.8

def test_group_duplicates_do_not_multiply_group_evidence():
    a=summary([.9,.7,.1,.2],[1,0,0,0],.6,['p','n','n','m'])
    assert a['false_positive']==1
    assert a['false_alarms_per_10000_negatives']==pytest.approx(10000/3)
    assert a['negative_time_groups']['groups']==2
    assert a['negative_time_groups']['groups_with_any_false_alarm']==1

def test_auroc_and_ap_handle_ties():
    a=summary([.5,.5,.5,.5],[1,1,0,0],.6)
    assert a['auroc']==.5
    assert a['average_precision']==.5
    assert a['precision']is None

def test_rare_search_counts_recall_not_merely_existing_targets():
    a=projected_search(1/96,9/96)
    assert a['targets_present']==100
    assert a['expected_recovered_targets']==pytest.approx(100/96)
    assert a['expected_false_alarms']==pytest.approx(93740.625)

def test_calibration_rejects_nonfinite_scores():
    with pytest.raises(ValueError):calibration_cutoff([0,np.nan])
