import math
from unittest.mock import patch

import pandas as pd
import pytest

from mta import MTA, MTAConfig


@pytest.fixture
def markov_mta():
    data = pd.DataFrame(
        {
            "path": ["a > b", "a > c"],
            "total_conversions": [3, 1],
            "total_null": [1, 1],
        }
    )
    return MTA(data, add_timepoints=False, normalize_by_default=False)


def test_transition_matrix_contains_expected_probabilities(markov_mta):
    transitions = markov_mta.transition_matrix()

    assert transitions[(markov_mta.START, "a")] == 1
    assert transitions[("a", "b")] == pytest.approx(2 / 3)
    assert transitions[("a", "c")] == pytest.approx(1 / 3)
    assert transitions[("b", markov_mta.CONV)] == pytest.approx(3 / 4)
    assert transitions[("b", markov_mta.NULL)] == pytest.approx(1 / 4)
    assert transitions[("c", markov_mta.CONV)] == pytest.approx(1 / 2)
    assert transitions[("c", markov_mta.NULL)] == pytest.approx(1 / 2)

    for channel in [markov_mta.START, "a", "b", "c"]:
        outgoing = sum(
            probability
            for (source, _), probability in transitions.items()
            if source == channel
        )
        assert outgoing == pytest.approx(1)


def test_analytical_markov_returns_all_channels_and_normalizes(markov_mta):
    markov_mta.markov(sim=False, normalize=True)

    result = markov_mta.attribution["markov"]
    assert set(result) == set(markov_mta.channels)
    assert all(value >= 0 for value in result.values())
    assert sum(result.values()) == pytest.approx(1, abs=1e-5)


def test_simulated_markov_uses_conversion_removal_effects():
    data = pd.DataFrame(
        {
            "path": ["a > b"],
            "total_conversions": [1],
            "total_null": [1],
        }
    )
    model = MTA(data, add_timepoints=False)
    outcomes = [
        {model.CONV: 80, model.NULL: 20},
        {model.CONV: 20, model.NULL: 80},
        {model.CONV: 40, model.NULL: 60},
    ]

    with patch.object(model, "simulate_path", side_effect=outcomes) as simulate:
        model.markov(sim=True, normalize=False)

    assert simulate.call_count == 3
    assert model.attribution["markov"] == {"a": 0.75, "b": 0.5}


def test_shao_single_channel_scores_are_hand_checkable():
    data = pd.DataFrame(
        {
            "path": ["a", "b"],
            "total_conversions": [3, 1],
            "total_null": [1, 3],
        }
    )
    model = MTA(data, add_timepoints=False, normalize_by_default=False)

    model.shao()

    assert model.attribution["shao"] == {"a": 2.25, "b": 0.25}


def test_logistic_regression_is_deterministic_and_normalized():
    data = pd.DataFrame(
        {
            "path": ["a", "b", "a > b", "a > c", "b > c"],
            "total_conversions": [20, 5, 15, 18, 8],
            "total_null": [5, 20, 15, 7, 17],
        }
    )
    model = MTA(data, add_timepoints=False)

    model.logistic_regression(
        test_size=0.25,
        sample_rows=1,
        sample_features=1,
        n_iterations=3,
    )

    result = model.attribution["linreg"]
    assert set(result) == {"a", "b", "c"}
    assert all(value >= 0 for value in result.values())
    assert sum(result.values()) == pytest.approx(1, abs=1e-5)


def test_additive_hazard_uses_time_order_and_normalizes():
    data = pd.DataFrame(
        {
            "path": ["a > b"],
            "exposure_times": ["2024-01-01 00:00:00 > 2024-01-01 00:00:01"],
            "total_conversions": [2],
            "total_null": [0],
        }
    )
    model = MTA(data, config=MTAConfig(allow_loops=True))

    contribution = model.pi(
        model.data.iloc[0]["path"],
        model.data.iloc[0]["exposure_times"],
        True,
        {"a": 1, "b": 1},
        {"a": 1, "b": 1},
    )
    assert contribution["b"] > contribution["a"]
    assert sum(contribution.values()) == pytest.approx(1)

    with patch("mta.mta.random.uniform", return_value=0.5):
        model.additive_hazard(epochs=0)

    assert sum(model.attribution["add_haz"].values()) == pytest.approx(1, abs=1e-5)


def _lr_data(conversions, nulls, n_rows=None, channels=("a", "b")):
    """Journeys with enough rows for a stratified bootstrap split."""
    n_rows = n_rows or len(conversions)
    # Vary the channel pair so remove_loops cannot merge rows into one.
    paths = [f"{channels[i % len(channels)]} > tail{i}" for i in range(n_rows)]
    return pd.DataFrame(
        {
            "path": paths,
            "total_conversions": conversions,
            "total_null": nulls,
        }
    )


def test_logistic_regression_rejects_fractional_counts():
    """Regression test: range() raised an opaque TypeError on float counts.

    The model expands each journey into one record per conversion/null, so a
    fractional count must be reported as a clear ValueError instead.
    Distinct channels keep the rows separate so the fraction survives the
    loop-removal aggregation step.
    """
    data = _lr_data([2.5] * 20, [1.5] * 20, channels=["a", "b", "c"])
    model = MTA(data, config=MTAConfig(add_timepoints=False, allow_loops=True))

    with pytest.raises(ValueError, match="whole-number counts"):
        model.logistic_regression(n_iterations=2)


def test_logistic_regression_accepts_whole_number_floats():
    """CSV columns read as float64 even when every value is integral."""
    data = _lr_data([3.0] * 20, [2.0] * 20)
    model = MTA(data, add_timepoints=False)

    model.logistic_regression(n_iterations=2)

    assert model.attribution["linreg"]
    assert all(math.isfinite(v) for v in model.attribution["linreg"].values())


def test_logistic_regression_tolerates_nan_counts():
    data = _lr_data([3.0] * 19 + [None], [2.0] * 20)
    model = MTA(data, add_timepoints=False)

    model.logistic_regression(n_iterations=2)

    assert model.attribution["linreg"]
    assert all(math.isfinite(v) for v in model.attribution["linreg"].values())


def test_fractional_counts_remain_valid_for_other_models():
    """Validation is scoped to logistic_regression; aggregation models cope."""
    data = pd.DataFrame(
        {
            "path": ["a > b", "c"],
            "total_conversions": [2.5, 1.5],
            "total_null": [1.5, 0.5],
        }
    )
    model = MTA(
        data, config=MTAConfig(add_timepoints=False, normalize_by_default=False)
    )

    model.first_touch(normalize=False).last_touch(normalize=False)

    assert model.attribution["first_touch"] == {"a": 2.5, "c": 1.5}
    assert model.attribution["last_touch"] == {"b": 2.5, "c": 1.5}


def test_empty_dataset_is_accepted_and_yields_empty_attribution():
    """Regression test: remove_loops crashed with KeyError('path').

    Every model already returns empty attribution for empty input, so the
    constructor must tolerate it and preserve the column layout.
    """
    data = pd.DataFrame(
        {
            "path": pd.Series([], dtype=object),
            "total_conversions": pd.Series([], dtype=float),
            "total_null": pd.Series([], dtype=float),
        }
    )

    model = MTA(data, add_timepoints=False)

    assert len(model.data) == 0
    assert list(model.data.columns) == ["path", "total_conversions", "total_null"]

    model.linear().first_touch().last_touch().position_based()
    model.time_decay().shao().shapley().markov(sim=False)

    assert all(v == {} for v in model.attribution.values())


@pytest.mark.parametrize("n_per_class", [1, 2, 3, 4, 5, 6])
def test_logistic_regression_survives_small_class_sizes(n_per_class):
    """Regression test: sklearn split/solver errors on tiny datasets.

    The pooled subsample could discard an entire class, and the stratified
    split needed more members than a small bootstrap provides.
    """
    data = _lr_data([1] * n_per_class, [1] * n_per_class, n_rows=n_per_class)

    model = MTA(data, config=MTAConfig(add_timepoints=False, allow_loops=True))
    model.logistic_regression(n_iterations=3)

    assert set(model.attribution["linreg"]) <= set(model.channels)


def test_logistic_regression_reports_single_class_data():
    data = _lr_data([2] * 6, [0] * 6)

    model = MTA(data, config=MTAConfig(add_timepoints=False, allow_loops=True))

    with pytest.raises(ValueError, match="needs both converting and non-converting"):
        model.logistic_regression(n_iterations=2)


def test_logistic_regression_reports_no_records():
    data = pd.DataFrame(
        {"path": ["a > b"], "total_conversions": [0], "total_null": [0]}
    )

    model = MTA(data, add_timepoints=False)

    with pytest.raises(ValueError, match="no records to fit"):
        model.logistic_regression(n_iterations=2)
