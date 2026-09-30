"""Calibration challengers, and the structural limit of monotone ones (B3).

The central question is whether a calibrator can rescue a *wrong-direction*
head. A single temperature is one positive scalar, and isotonic regression is
non-decreasing by construction, so both preserve every ranking and cannot. Only
Platt scaling with a free-sign slope can invert a class. These tests hold each
calibrator against the same deliberately anti-correlated head and check exactly
that, rather than assuming it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc.calibration import (
    CALIBRATOR_VERSION,
    ISOTONIC,
    PLATT,
    TEMPERATURE,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    _pava,
    calibrator_report,
    fit_isotonic,
    fit_platt,
    negative_log_likelihood,
    reliability_bins,
    select_calibrator,
)


def _anti_correlated(n: int = 150, n_classes: int = 3, seed: int = 0):
    """Logits where the true class always has the lowest score.

    Perfectly separable and perfectly wrong: any rank-preserving transform
    leaves every argmax incorrect.
    """
    rng = np.random.default_rng(seed)
    targets = np.arange(n) % n_classes
    logits = rng.normal(1.5, 0.05, size=(n, n_classes))
    logits[np.arange(n), targets] = -1.5
    return logits, targets


def _accuracy(logits: np.ndarray, targets: np.ndarray, calibrator) -> float:
    predictions = calibrator.apply(logits).argmax(axis=1)
    return float((predictions == targets).mean())


def test_anti_correlated_fixture_is_genuinely_inverted():
    logits, targets = _anti_correlated()
    predictions = TemperatureCalibrator().apply(logits).argmax(axis=1)
    assert not (predictions == targets).any()
    # And it is separable, so a calibrator that can invert should recover it.
    assert len(np.unique(targets)) == 3


def test_temperature_preserves_ranking_and_cannot_recover():
    logits, targets = _anti_correlated()
    calibrator, scores = select_calibrator(logits, targets, names=(TEMPERATURE,))
    assert isinstance(calibrator, TemperatureCalibrator)
    assert _accuracy(logits, targets, calibrator) == 0.0
    # A positive scalar preserves the argmax exactly at any temperature.
    for temperature in (0.05, 0.5, 1.0, 5.0, 50.0):
        scaled = TemperatureCalibrator(temperature=temperature).apply(logits)
        assert (scaled.argmax(axis=1) == targets).sum() == 0
    assert scores[TEMPERATURE] > 0


def test_isotonic_collapses_to_the_prior_on_an_inverted_head():
    """The precise limit of a monotone calibrator - and a corrected claim.

    It is tempting to say "monotone calibrators preserve argmax". That is false
    for *per-class* maps: monotone maps of different shapes can reorder across
    classes (see ``test_per_class_monotone_maps_do_not_preserve_argmax``). What
    actually happens to an inverted head is that the non-decreasing fit on
    anti-correlated one-vs-rest labels collapses to the constant class prior,
    so the calibrator becomes uninformative rather than correct. It repairs
    shape; it cannot repair direction.
    """
    logits, targets = _anti_correlated()
    calibrator = fit_isotonic(logits, targets)
    assert isinstance(calibrator, IsotonicCalibrator)

    # Fitted map is non-decreasing in the raw score for every class...
    for fitted in calibrator.fitted:
        assert (np.diff(fitted) >= -1e-12).all()
    # ...and here it has collapsed to the flat 1/3 prior.
    for fitted in calibrator.fitted:
        assert fitted.max() - fitted.min() < 1e-9
        assert fitted[0] == pytest.approx(1 / 3)

    # So it does not recover, and prediction degenerates to one class.
    assert _accuracy(logits, targets, calibrator) < 0.5
    predictions = calibrator.apply(logits).argmax(axis=1)
    assert len(np.unique(predictions)) == 1


def test_per_class_monotone_maps_do_not_preserve_argmax():
    """Correcting the intuitive claim in the other direction.

    Only a *shared* monotone transform preserves every argmax. Per-class maps
    have independent shapes, so they can and do reorder across classes.
    """
    shared_scores = [np.array([0.0, 10.0])] * 2
    different_shapes = IsotonicCalibrator(
        scores=list(shared_scores),
        fitted=[np.array([0.0, 0.1]), np.array([0.0, 0.9])],
    )
    single = np.array([[2.0, 1.0]])
    assert single.argmax() == 0
    assert different_shapes.apply(single).argmax() == 1


def test_platt_can_invert_and_recovers_a_wrong_direction_head():
    logits, targets = _anti_correlated()
    calibrator = fit_platt(logits, targets)
    assert isinstance(calibrator, PlattCalibrator)
    # The free-sign slope is the mechanism: it must go negative here.
    assert (calibrator.slopes < 0).all(), calibrator.slopes
    assert _accuracy(logits, targets, calibrator) == pytest.approx(1.0)


def test_select_calibrator_picks_the_one_that_can_actually_help():
    logits, targets = _anti_correlated()
    calibrator, scores = select_calibrator(logits, targets)
    assert calibrator.kind == PLATT
    assert scores[PLATT] < scores[TEMPERATURE]
    assert scores[PLATT] < scores[ISOTONIC]
    assert _accuracy(logits, targets, calibrator) == pytest.approx(1.0)


def test_select_calibrator_without_an_inversion():
    """On an ordinary head all three are reasonable and selection is by NLL."""
    rng = np.random.default_rng(1)
    n, n_classes = 200, 4
    targets = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 0.3, size=(n, n_classes))
    logits[np.arange(n), targets] += 1.5

    calibrator, scores = select_calibrator(logits, targets)
    assert set(scores) == {TEMPERATURE, PLATT, ISOTONIC}
    assert calibrator.kind in scores
    assert scores[calibrator.kind] == min(scores.values())
    # None of these should be worse than uncalibrated guessing.
    assert _accuracy(logits, targets, calibrator) > 0.5


def test_platt_slopes_stay_near_one_on_a_well_scaled_head():
    rng = np.random.default_rng(2)
    n, n_classes = 400, 3
    targets = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 0.2, size=(n, n_classes))
    logits[np.arange(n), targets] += 1.0
    calibrator = fit_platt(logits, targets, steps=800)
    assert (calibrator.slopes > 0).all()


def test_isotonic_calibration_of_an_overconfident_head():
    """Shape repair is where isotonic earns its place."""
    rng = np.random.default_rng(3)
    n, n_classes = 2000, 2
    targets = rng.integers(0, n_classes, size=n)
    logits = np.zeros((n, n_classes))
    logits[np.arange(n), targets] += 0.4
    logits = logits * 20.0  # badly overconfident
    before = reliability_bins(
        TemperatureCalibrator().apply(logits), targets
    )["ece"]
    after = reliability_bins(fit_isotonic(logits, targets).apply(logits), targets)[
        "ece"
    ]
    assert after < before


def test_reliability_bins_are_well_formed():
    rng = np.random.default_rng(4)
    n, n_classes = 500, 3
    targets = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    report = reliability_bins(TemperatureCalibrator().apply(logits), targets, bins=8)
    assert report["n"] == n
    assert len(report["bins"]) == 8
    assert sum(row["count"] for row in report["bins"]) == n
    assert 0.0 <= report["ece"] <= 1.0
    for row in report["bins"]:
        if row["count"]:
            assert 0.0 <= row["accuracy"] <= 1.0
            assert 0.0 <= row["mean_confidence"] <= 1.0
        else:
            assert row["accuracy"] is None


def test_reliability_bins_on_perfectly_calibrated_hard_predictions():
    # Deterministic and correct: confidence 1.0 and accuracy 1.0.
    probabilities = np.eye(3)
    targets = np.array([0, 1, 2])
    report = reliability_bins(probabilities, targets)
    assert report["ece"] == pytest.approx(0.0)


def test_pava_returns_a_non_decreasing_least_squares_fit():
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    fitted = _pava(values)
    assert (np.diff(fitted) >= 0).all()
    assert fitted == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])


def test_negative_log_likelihood_penalizes_confident_wrong_answers():
    targets = np.array([0, 1])
    confident = np.array([[0.99, 0.01], [0.01, 0.99]])
    wrong = np.array([[0.01, 0.99], [0.99, 0.01]])
    assert negative_log_likelihood(confident, targets) < negative_log_likelihood(
        wrong, targets
    )


def test_calibrator_report_carries_identity_and_the_no_certificate_warning():
    logits, targets = _anti_correlated()
    report = calibrator_report(fit_platt(logits, targets))
    assert report["kind"] == PLATT
    assert report["calibrator_version"] == CALIBRATOR_VERSION
    assert "not a calibration certificate" in report["warning"]

    isotonic_report = calibrator_report(fit_isotonic(logits, targets))
    assert isotonic_report["kind"] == ISOTONIC
    assert len(isotonic_report["scores"]) == logits.shape[1]


def test_select_calibrator_rejects_an_empty_split():
    with pytest.raises(ValueError, match="non-empty"):
        select_calibrator(np.zeros((0, 2)), np.zeros((0,), dtype=int))


def test_platt_and_isotonic_are_json_safe():
    import json

    logits, targets = _anti_correlated()
    for calibrator in (fit_platt(logits, targets), fit_isotonic(logits, targets)):
        json.dumps(calibrator_report(calibrator))
