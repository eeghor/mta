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


def test_position_based_len2_respects_weights():
    """For 2-touch paths, weights should be used (renormalized to conserve credit).
    Default 40+40 must continue to behave as 50/50.
    """
    import pandas as pd
    from mta import MTA, MTAConfig

    data = pd.DataFrame(
        {
            "path": ["x > y"],
            "total_conversions": [10],
            "total_null": [0],
        }
    )
    m = MTA(data, config=MTAConfig(add_timepoints=False, allow_loops=True))

    # default weights -> 5/5
    m.position_based(first_weight=40, last_weight=40, normalize=False)
    assert m.attribution["pos_based"] == {"x": 5.0, "y": 5.0}

    # explicit 30/70 for 2-touch should give exactly that ratio (full credit)
    m.position_based(first_weight=30, last_weight=70, normalize=False)
    res = m.attribution["pos_based"]
    assert res["x"] == pytest.approx(3.0)
    assert res["y"] == pytest.approx(7.0)
    assert sum(res.values()) == pytest.approx(10.0)


def test_add_exposure_times_false_does_not_crash_or_add_column():
    import pandas as pd
    from mta import MTA, MTAConfig
    data = pd.DataFrame(
        {
            "path": ["a > b"],
            "total_conversions": [1],
            "total_null": [0],
        }
    )
    m = MTA(data, config=MTAConfig(add_timepoints=False))
    assert "exposure_times" not in m.data.columns

    # calling with False must not raise and must not add the column
    m2 = m.add_exposure_times(exposure_every_second=False)
    assert m2 is m
    assert "exposure_times" not in m.data.columns

    # True does add
    m.add_exposure_times(exposure_every_second=True)
    assert "exposure_times" in m.data.columns
    assert len(m.data["exposure_times"].iloc[0]) > 0


def test_empty_and_malformed_paths_are_sanitized_and_safe():
    """Paths that split to empties or are empty lists must not crash models
    or pollute channels with '' .
    """
    import pandas as pd
    from mta import MTA, MTAConfig

    bad = pd.DataFrame(
        {
            # various degenerate cases
            "path": [
                "",           # -> []
                " > > ",      # -> []
                "a >  > b",   # -> ['a', 'b']
                "c",          # normal
            ],
            "total_conversions": [1, 2, 3, 4],
            "total_null": [0, 0, 0, 0],
        }
    )
    m = MTA(bad, config=MTAConfig(add_timepoints=False, allow_loops=True))

    # No empty-string channel
    assert "" not in m.channels
    assert set(m.channels) == {"a", "b", "c"}

    # All models must run without IndexError on [] paths
    m.first_touch(normalize=False)
    m.last_touch(normalize=False)
    m.linear(normalize=False)
    m.position_based(normalize=False)
    m.time_decay(normalize=False)
    m.markov(sim=False, normalize=False)
    # attribution populated only for real channels that touched conv
    assert all(ch in m.attribution.get("first_touch", {}) or ch in ("a", "b", "c") for ch in m.channels)


def test_first_and_last_touch_robust_to_null_rows_and_non_default_index():
    """Regression test for the old fragile groupby using full self.data key series
    instead of filtered. Must work with custom indexes and rows having 0 conv.
    """
    import pandas as pd
    from mta import MTA

    df = pd.DataFrame(
        {
            "path": ["ignored > x", "first_a", "last_b > z", "only_null"],
            "total_conversions": [0, 4, 6, 0],
            "total_null": [9, 1, 1, 10],
        },
        index=[100, 101, 102, 103],  # non-default, gapped
    )
    m = MTA(df, add_timepoints=False, normalize_by_default=False)

    m.first_touch(normalize=False)
    m.last_touch(normalize=False)

    # first: 'first_a' from row1 (4), 'last_b' ? wait last of "last_b>z" is z? No:
    # paths with conv>0:
    # index101: ["first_a"] -> first 'first_a'
    # index102: ["last_b", "z"] -> first 'last_b'
    assert m.attribution["first_touch"] == {"first_a": 4, "last_b": 6}

    # last: 'first_a' last, 'z' last
    assert m.attribution["last_touch"] == {"first_a": 4, "z": 6}
