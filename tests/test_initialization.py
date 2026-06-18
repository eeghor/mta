import pandas as pd
import pytest

from mta import MTA, MTAConfig


@pytest.mark.parametrize("missing_column", ["path", "total_conversions", "total_null"])
def test_required_columns_are_validated(missing_column):
    data = pd.DataFrame(
        {
            "path": ["a > b"],
            "total_conversions": [1],
            "total_null": [0],
        }
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match="Data must contain columns"):
        MTA(data)


@pytest.mark.parametrize("invalid_data", [None, 42, ["a", "b"]])
def test_unsupported_input_types_are_rejected(invalid_data):
    with pytest.raises(TypeError, match="file path or pandas DataFrame"):
        MTA(invalid_data)


def test_dataframe_input_is_copied(journey_data):
    original = journey_data.copy(deep=True)

    MTA(journey_data)

    pd.testing.assert_frame_equal(journey_data, original)


def test_keyword_configuration_and_repr(journey_data):
    model = MTA(
        journey_data,
        add_timepoints=False,
        allow_loops=True,
        sep=" > ",
        normalize_by_default=False,
    )

    assert model.config == MTAConfig(
        add_timepoints=False,
        allow_loops=True,
        sep=" > ",
        normalize_by_default=False,
    )
    assert repr(model) == "MTA(channels=4, journeys=3)"


def test_channel_indices_are_bijective(raw_mta):
    assert raw_mta.channels == ["direct", "email", "search", "social"]
    assert raw_mta.channels_ext[0] == raw_mta.START
    assert raw_mta.channels_ext[-2:] == [raw_mta.CONV, raw_mta.NULL]

    for channel, index in raw_mta.channel_name_to_index.items():
        assert raw_mta.index_to_channel_name[index] == channel
