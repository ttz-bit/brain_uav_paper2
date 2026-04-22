from scripts.prepare_seadronessee import build_split_leakage_report


def test_split_leakage_report_detects_overlap():
    report = build_split_leakage_report(
        {
            "train": {"1", "2", "3"},
            "val": {"4", "5"},
            "test": {"2", "9"},
        }
    )
    assert report["all_clear"] is False
    assert any(p["overlap_count"] > 0 for p in report["pairs"])


def test_split_leakage_report_clear_when_disjoint():
    report = build_split_leakage_report(
        {
            "train": {"1", "2", "3"},
            "val": {"4", "5"},
            "test": {"9"},
        }
    )
    assert report["all_clear"] is True
