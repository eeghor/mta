import pytest


def assert_credit_is_conserved(attribution, expected_total=10):
    assert sum(attribution.values()) == pytest.approx(expected_total)


def test_first_and_last_touch_have_exact_allocations(raw_mta):
    returned = raw_mta.first_touch(normalize=False).last_touch(normalize=False)

    assert returned is raw_mta
    assert raw_mta.attribution["first_touch"] == {"direct": 2, "search": 8}
    assert raw_mta.attribution["last_touch"] == {"direct": 2, "email": 8}


def test_linear_equal_share_has_exact_allocation(raw_mta):
    raw_mta.linear(share="same", normalize=False)

    result = raw_mta.attribution["linear"]
    assert result == pytest.approx(
        {
            "direct": 2,
            "search": 19 / 6,
            "email": 19 / 6,
            "social": 5 / 3,
        }
    )
    assert_credit_is_conserved(result)


def test_linear_proportional_counts_repeated_touches():
    import pandas as pd

    from mta import MTA, MTAConfig

    model = MTA(
        pd.DataFrame(
            {
                "path": ["a > a > b"],
                "total_conversions": [6],
                "total_null": [0],
            }
        ),
        config=MTAConfig(allow_loops=True, add_timepoints=False),
    )

    model.linear(share="proportional", normalize=False)

    assert model.attribution["linear"] == {"a": 4, "b": 2}


def test_position_based_has_exact_allocation(raw_mta):
    raw_mta.position_based(first_weight=40, last_weight=40, normalize=False)

    result = raw_mta.attribution["pos_based"]
    assert result == pytest.approx(
        {"direct": 2, "search": 3.5, "email": 3.5, "social": 1}
    )
    assert_credit_is_conserved(result)


def test_time_decay_favors_later_unique_touches(raw_mta):
    raw_mta.time_decay(count_direction="left", normalize=False)

    result = raw_mta.attribution["time_decay"]
    assert result["email"] > result["search"] > result["social"]
    assert_credit_is_conserved(result)


@pytest.mark.parametrize(
    ("method", "kwargs", "message"),
    [
        ("linear", {"share": "unknown"}, "share must be"),
        (
            "position_based",
            {"first_weight": 60, "last_weight": 50},
            "cannot exceed 100",
        ),
        ("time_decay", {"count_direction": "middle"}, "count_direction must be"),
    ],
)
def test_invalid_model_options_are_rejected(raw_mta, method, kwargs, message):
    with pytest.raises(ValueError, match=message):
        getattr(raw_mta, method)(**kwargs)


@pytest.mark.parametrize(
    "method_name",
    ["linear", "position_based", "time_decay", "first_touch", "last_touch"],
)
def test_heuristic_models_normalize_to_one(raw_mta, method_name):
    getattr(raw_mta, method_name)(normalize=True)

    result_key = "pos_based" if method_name == "position_based" else method_name
    assert sum(raw_mta.attribution[result_key].values()) == pytest.approx(1, abs=1e-5)
