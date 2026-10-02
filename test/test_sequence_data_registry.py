"""Sequence databases and protein language models: their registry, and the
resumable, checksummed download (against a local HTTP server, offline)."""
import hashlib
import http.server
import json
import os
import threading

import pytest

import IMP.bff as bff

REGISTRY = os.path.join(os.path.dirname(__file__), "..", "data", "sequence_registry.json")


def test_the_registry_is_well_formed():
    entries = bff.get_sequence_data_registry()
    raw = {k: v for k, v in json.load(open(REGISTRY)).items() if not k.startswith("_")}
    assert entries == raw
    for name in ("uniref_consurf", "uniprot_sprot", "esm2_35m", "esm2_650m_contacts"):
        e = entries[name]
        assert e["path"].startswith(("sequence_databases/", "protein_language_model/"))
        assert e["bytes"] > 0 and e["description"] and e["source"]
        assert e["sha256"] == "PENDING" or (len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0)
    # never part of the small registry's all-fetch
    small = bff.get_data_registry()
    assert not any(k.startswith(("sequence_databases/", "protein_language_model/")) for k in small)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with byte-range support; `fail_after` cuts a response short once."""
    fail_after = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if _RangeHandler.fail_after is not None:
            n, _RangeHandler.fail_after = _RangeHandler.fail_after, None
            self.wfile.write(body[:n])
            self.wfile.flush()
            self.connection.shutdown(2)          # the connection drops mid-file
            return
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "served"
    (root / "sequence_databases").mkdir(parents=True)
    payload = os.urandom(3 * (1 << 20) + 12345)
    (root / "sequence_databases" / "toy.pto").write_bytes(payload)
    handler = lambda *a, **k: _RangeHandler(*a, directory=str(root), **k)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/" % httpd.server_address[1], payload
    httpd.shutdown()


def _toy_registry(payload, sha=None):
    return {"toy": {"path": "sequence_databases/toy.pto", "bytes": len(payload),
                    "sha256": sha or hashlib.sha256(payload).hexdigest(),
                    "description": "test", "source": "test"}}


def test_download_resumes_and_checks(server, tmp_path, monkeypatch):
    url, payload = server
    monkeypatch.setattr(bff, "DATA_URL", url)
    monkeypatch.setattr(bff, "get_sequence_data_registry", lambda: _toy_registry(payload))
    _RangeHandler.fail_after = 1 << 20                 # first attempt dies after 1 MB
    out = bff.fetch_sequence_data("toy", str(tmp_path / "dest"))
    assert open(out, "rb").read() == payload
    assert not os.path.exists(out + ".part")
    # present and verified: no second download (the server is not asked)
    monkeypatch.setattr(bff, "DATA_URL", "http://127.0.0.1:9/")
    assert bff.fetch_sequence_data("toy", str(tmp_path / "dest")) == out


def test_a_wrong_checksum_is_refused_and_unpublished_entries_too(server, tmp_path, monkeypatch):
    url, payload = server
    monkeypatch.setattr(bff, "DATA_URL", url)
    monkeypatch.setattr(bff, "get_sequence_data_registry", lambda: _toy_registry(payload, "0" * 64))
    with pytest.raises(bff.IOException):
        bff.fetch_sequence_data("toy", str(tmp_path / "dest"))
    assert not os.path.exists(tmp_path / "dest" / "sequence_databases" / "toy.pto.part")
    monkeypatch.setattr(bff, "get_sequence_data_registry", lambda: _toy_registry(payload, "PENDING"))
    with pytest.raises(bff.IOException):
        bff.fetch_sequence_data("toy", str(tmp_path / "dest"))
    with pytest.raises(bff.IOException):
        bff.fetch_sequence_data("no_such_database")
