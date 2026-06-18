import pandas as pd
import pytest

from mta import MTA, MTAConfig


@pytest.fixture
def journey_data():
    """A small dataset whose total conversion credit is easy to verify."""
    return pd.DataFrame(
        {
            "path": ["direct", "search > email", "search > social > email"],
            "total_conversions": [2, 3, 5],
            "total_null": [1, 2, 4],
        }
    )


@pytest.fixture
def raw_mta(journey_data):
    return MTA(
        journey_data,
        config=MTAConfig(
            add_timepoints=False,
            allow_loops=True,
            normalize_by_default=False,
        ),
    )
