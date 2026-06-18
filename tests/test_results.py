import json

import pandas as pd
import pytest


def test_show_returns_sorted_percentages(raw_mta, capsys):
    raw_mta.first_touch(normalize=True)

    result = raw_mta.show()

    assert result.index.tolist() == ["direct", "search"]
    assert result["first_touch"].tolist() == [20.0, 80.0]
    assert "Attribution Results (%)" in capsys.readouterr().out


def test_show_can_filter_channels(raw_mta):
    raw_mta.linear(normalize=True)

    result = raw_mta.show(channels=["email", "social"])

    assert result.index.tolist() == ["email", "social"]


def test_export_csv_preserves_raw_values(raw_mta, tmp_path):
    raw_mta.first_touch(normalize=False)
    output = tmp_path / "attribution.csv"

    raw_mta.export_results(str(output), format="csv", as_percentage=False)

    exported = pd.read_csv(output, index_col=0)
    assert exported.loc["direct", "first_touch"] == 2
    assert exported.loc["search", "first_touch"] == 8


def test_export_json_can_write_percentages(raw_mta, tmp_path):
    raw_mta.first_touch(normalize=True)
    output = tmp_path / "attribution.json"

    raw_mta.export_results(str(output), format="json", as_percentage=True)

    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported == {
        "direct": {"first_touch": 20.0},
        "search": {"first_touch": 80.0},
    }


def test_export_rejects_unknown_format(raw_mta, tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        raw_mta.export_results(str(tmp_path / "result.bin"), format="binary")


def test_normalize_dict_handles_rounding_and_zero_totals(raw_mta):
    assert raw_mta.normalize_dict({"a": 1, "b": 2}) == {
        "a": 0.333333,
        "b": 0.666667,
    }
    assert raw_mta.normalize_dict({"a": 0, "b": 0}) == {"a": 0, "b": 0}
