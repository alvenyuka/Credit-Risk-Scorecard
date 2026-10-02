"""The relational-feature cache is used only when its fingerprint matches.

The fingerprint covers the aggregation code and the size of every input table, so
replacing a CSV changes it and forces a rebuild instead of a silent read.
"""
from relational_features import SOURCE_TABLES, cache_fingerprint


def _fake_data_dir(tmp_path, sizes):
    for name, size in zip(SOURCE_TABLES, sizes):
        (tmp_path / name).write_bytes(b"x" * size)
    return tmp_path


def test_fingerprint_is_stable_for_unchanged_inputs(tmp_path):
    d = _fake_data_dir(tmp_path, range(1, len(SOURCE_TABLES) + 1))
    assert cache_fingerprint(d) == cache_fingerprint(d)


def test_fingerprint_changes_when_an_input_table_changes(tmp_path):
    d = _fake_data_dir(tmp_path, range(1, len(SOURCE_TABLES) + 1))
    before = cache_fingerprint(d)
    (d / SOURCE_TABLES[0]).write_bytes(b"x" * 999)
    assert cache_fingerprint(d) != before


def test_fingerprint_changes_when_a_table_is_missing(tmp_path):
    d = _fake_data_dir(tmp_path, range(1, len(SOURCE_TABLES) + 1))
    before = cache_fingerprint(d)
    (d / SOURCE_TABLES[-1]).unlink()
    assert cache_fingerprint(d) != before
