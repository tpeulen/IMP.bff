"""The settings file: where databases and servers are on this machine."""

import json

import pytest

import IMP.bff as bff


def _write(tmp_path, doc, name="settings.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def test_the_path_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("IMP_BFF_SETTINGS", str(tmp_path / "mine.json"))
    assert bff.get_settings_path() == str(tmp_path / "mine.json")
    monkeypatch.delenv("IMP_BFF_SETTINGS")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert bff.get_settings_path() == str(tmp_path / "imp.bff" / "settings.json")


def test_a_missing_file_or_section_is_the_defaults(tmp_path):
    s = bff.get_sequence_search_settings(str(tmp_path / "absent.json"))
    assert list(s.get_database_names()) == [] and s.default_database == ""
    assert s.threads == 0 and s.memory_budget_mb == 2048
    s = bff.get_sequence_search_settings(_write(tmp_path, {"other": {"x": 1}}))
    assert list(s.get_server_names()) == []


def test_a_full_section_reads_and_relative_paths_follow_the_file(tmp_path):
    path = _write(tmp_path, {"sequence_search": {
        "databases": {"uniref90": "/Volumes/SD1TB/uniref90.pto", "small": "db/small.pto"},
        "default_database": "uniref90",
        "indexes": {"uniref90": "~/idx.pto"},
        "servers": {"colabfold": {"url": "https://example.org", "poll_seconds": 2}},
        "fallback_server": "colabfold",
        "threads": 6,
        "memory_budget_mb": 512}})
    s = bff.get_sequence_search_settings(path)
    assert list(s.get_database_names()) == ["small", "uniref90"]
    assert s.get_database("uniref90") == "/Volumes/SD1TB/uniref90.pto"
    assert s.get_database("small") == str(tmp_path / "db" / "small.pto")
    assert s.get_index("uniref90").endswith("/idx.pto") and not s.get_index("uniref90").startswith("~")
    assert s.get_index("small") == ""
    server = s.get_server("colabfold")
    assert server.url == "https://example.org" and server.protocol == "colabfold-v1"
    assert server.poll_seconds == 2 and server.timeout_seconds == 3600
    assert s.default_database == "uniref90" and s.fallback_server == "colabfold"
    assert s.threads == 6 and s.memory_budget_mb == 512
    with pytest.raises(bff.ValueException):
        s.get_database("nope")


@pytest.mark.parametrize("section", [
    {"databse": {}},                                       # misspelt key
    {"databases": {"a": 3}},                               # not a path
    {"databases": {}, "default_database": "a"},            # names nothing
    {"servers": {"s": {"protocol": "colabfold-v1"}}},      # no url
    {"servers": {"s": {"url": "u", "pol_seconds": 1}}},    # misspelt server key
    {"fallback_server": "s"},                              # names nothing
])
def test_mistakes_are_refused(tmp_path, section):
    with pytest.raises(bff.ValueException):
        bff.get_sequence_search_settings(_write(tmp_path, {"sequence_search": section}))


def test_invalid_json_is_refused(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(bff.ValueException):
        bff.get_sequence_search_settings(str(p))
