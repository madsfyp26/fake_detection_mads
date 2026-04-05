"""Unsupervised vs supervised AVH calibration must use different mappings."""

from calibration_runtime import (
    avh_score_to_calibrated_p_fake,
    avh_score_to_p_fake,
    avh_unsupervised_score_to_p_fake,
)


def test_unsup_not_identical_to_supervised_for_typical_unsup_score():
    s = 1.0
    ps = avh_score_to_p_fake(s)
    pu = avh_unsupervised_score_to_p_fake(s)
    assert abs(ps - pu) > 0.05, "supervised sigmoid should differ from unsupervised at same numeric score"


def test_dispatch_matches_branch():
    s = 0.8
    assert avh_score_to_calibrated_p_fake(s, use_unsup_avh=True) == avh_unsupervised_score_to_p_fake(s)
    assert avh_score_to_calibrated_p_fake(s, use_unsup_avh=False) == avh_score_to_p_fake(s)


def test_unsup_center_near_half():
    import calibration_runtime as cr

    c = float(cr.DEFAULTS["avh_unsup_center"])
    p = avh_unsupervised_score_to_p_fake(c)
    assert 0.45 <= p <= 0.55
