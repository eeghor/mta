import json

import pandas as pd
import pytest


def test_show_returns_sorted_percentages(raw_mta, capsys):
    raw_mta.first_touch(normalize=True)

    result = raw_mta.show()

    # show() now reindexes to all channels (with 0 credit for untouched)
    assert result.index.tolist() == ["direct", "email", "search", "social"]
    assert result["first_touch"].tolist() == [20.0, 0.0, 80.0, 0.0]
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
    # Includes 0-credit channels because of reindex+fillna for consistency
    assert exported == {
        "direct": {"first_touch": 20.0},
        "email": {"first_touch": 0.0},
        "search": {"first_touch": 80.0},
        "social": {"first_touch": 0.0},
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


def test_show_auto_detects_raw_vs_percent(raw_mta, capsys):
    """show() should *100 only for normalized (0-1) results, show raw otherwise.
    Prevents the bug where normalize=False produced 800 etc instead of 8.
    """
    # Raw
    raw_mta.first_touch(normalize=False)
    out_raw = raw_mta.show()
    captured = capsys.readouterr().out
    assert "Attribution Results:" in captured
    assert "Attribution Results (%)" not in captured
    # raw values preserved (sum=10 for this fixture)
    assert out_raw["first_touch"].sum() == pytest.approx(10.0)

    # Now normalized (recompute)
    raw_mta.first_touch(normalize=True)
    out_pct = raw_mta.show()
    captured = capsys.readouterr().out
    assert "Attribution Results (%)" in captured
    assert out_pct["first_touch"].sum() == pytest.approx(100.0)
    assert out_pct.loc["direct", "first_touch"] == 20.0


def test_show_and_export_fillna_zero_for_missing_channels(raw_mta):
    """Models that don't touch every channel (e.g. first_touch) should not produce
    NaNs in comparison tables / exports. fillna(0) in display layer.
    """
    raw_mta.first_touch(normalize=False)
    df = raw_mta.show()
    assert not df.isna().any().any()
    # 'email' etc never first-touch so 0 in display
    assert "email" in df.index
    assert df.loc["email", "first_touch"] == 0.0
