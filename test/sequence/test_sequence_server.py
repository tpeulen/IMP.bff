"""An MSA server as the fallback: A3M rows as hits, the colabfold-v1
protocol against a local stand-in server, and ConSurf through it."""

import json
import os
import subprocess
import sys

import pytest

import IMP.bff as bff

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "input", "sequence", "rbp_60x120.fasta")


def test_a3m_rows_become_hits():
    a3m = (">101\nACDEFGHIKL\n"
           ">hitA 120 0.800 1.5E-20 0 10 10 0 12 40\nACDxEFGHIK-\n"   # 'x': an insertion
           ">hitB\n--DEFGHI--\n")
    a3m = a3m.replace("ACDxEFGHIK-", "ACDxEFGHIKL")
    hits = bff.get_a3m_hits(a3m)
    assert [h.identifier for h in hits] == ["hitA", "hitB"]
    a, b = hits
    assert a.aligned == "ACDEFGHIKL" and a.evalue == 1.5e-20 and a.target_length == 40
    assert a.query_start == 0 and a.query_end == 10
    assert abs(a.identity - 10 / 11) < 1e-12          # the insertion is a column
    assert b.aligned == "--DEFGHI--" and b.evalue == 0 and b.identity == 1.0
    assert (b.query_start, b.query_end, b.target_end) == (2, 8, 6)
    with pytest.raises(bff.ValueException):
        bff.get_a3m_hits(">101\nACD\n>bad\nACDE\n")


def _family_a3m():
    msa = bff.read_sequence_msa(FIXTURE, match_columns_only=False)
    ref = msa.get_sequence(0)
    keep = [k for k, c in enumerate(ref) if c != "-"]      # the query's columns
    rows = [">101\n" + "".join(ref[k] for k in keep)]
    for s in range(1, msa.get_n_sequences()):
        seq = msa.get_sequence(s)
        rows.append(f">fam{s}\n" + "".join(seq[k] for k in keep))
    return "\n".join(rows) + "\n", "".join(ref[k] for k in keep)


# ColabFold's MMseqs2 API, in the part a client sees. In a process of its own:
# the library holds the interpreter's lock while it waits on the network.
SERVER = r"""
import gzip, io, json, sys, tarfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
a3m = open(sys.argv[1]).read()
polls = [0]

class Colabfold(BaseHTTPRequestHandler):
    def reply(self, body, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
        ok = form.get("mode") == ["all"] and form["q"][0].startswith(">101\n")
        self.reply(json.dumps({"id": "t1", "status": "PENDING" if ok else "ERROR"}).encode())

    def do_GET(self):
        if self.path == "/ticket/t1":
            polls[0] += 1
            open(sys.argv[2], "w").write(str(polls[0]))
            status = "COMPLETE" if polls[0] >= 2 else "RUNNING"
            self.reply(json.dumps({"id": "t1", "status": status}).encode())
        elif self.path == "/result/download/t1":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                for name, text in (("pdb70.m8", ""), ("uniref.a3m", a3m)):
                    data = text.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            self.reply(gzip.compress(buf.getvalue()), "application/gzip")
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

httpd = HTTPServer(("127.0.0.1", 0), Colabfold)
print(httpd.server_port, flush=True)
httpd.serve_forever()
"""


@pytest.fixture
def server(tmp_path, monkeypatch):
    if not bff.get_sequence_server_available():
        pytest.skip("built without libcurl")
    a3m, query = _family_a3m()
    (tmp_path / "uniref.a3m").write_text(a3m)
    (tmp_path / "server.py").write_text(SERVER)
    polls = tmp_path / "polls"
    proc = subprocess.Popen([sys.executable, str(tmp_path / "server.py"),
                             str(tmp_path / "uniref.a3m"), str(polls)],
                            stdout=subprocess.PIPE, text=True)
    port = int(proc.stdout.readline())
    url = f"http://127.0.0.1:{port}/"
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"sequence_search": {
        "servers": {"local": {"url": url, "poll_seconds": 0.01, "timeout_seconds": 30}},
        "fallback_server": "local"}}))
    monkeypatch.setenv("IMP_BFF_SETTINGS", str(settings))
    yield url, query, a3m, polls
    proc.kill()
    proc.wait()


def test_fetch_follows_the_ticket_and_unpacks_the_result(server):
    url, query, a3m, polls = server
    s = bff.get_sequence_search_settings().get_server("local")
    assert bff.fetch_server_msa(query, s) == a3m
    assert polls.read_text() == "2"


def test_consurf_falls_back_to_the_server(server):
    url, query, a3m, polls = server
    options = bff.ConsurfOptions()
    options.homologs = bff.SequenceHomologOptions.consurf_standalone()
    r = bff.compute_consurf([query], options)[0]
    assert r.get_is_ok(), r.status
    assert len(r.homologs) >= 5 and all(h.identifier.startswith("fam") for h in r.homologs)
    assert r.conservation.get_n_positions() == len(query)


def test_an_unknown_protocol_is_refused():
    s = bff.SequenceSearchServer()
    s.name, s.url, s.protocol = "x", "http://127.0.0.1:9", "hhblits-v0"
    with pytest.raises(bff.ValueException):
        bff.fetch_server_msa("ACDEFG", s)
