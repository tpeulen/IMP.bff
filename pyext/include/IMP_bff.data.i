/*
 * Where the large data lives (PRD-137 step 6d): not in git, not in a wheel.
 *
 * data/ in the repository holds the small, always-needed files and
 * data/registry.json -- the path and sha256 of every file of
 * rotamer_library/ and cgprobe/ (62 MB), which are served from DATA_URL
 * (utility/data_registry.py --fetch restores them into a checkout). A wheel
 * ships the small set beside the module; a conda package ships everything
 * (its build fetches first). At run time get_data_path() looks where the
 * build installed, then in the fetch cache, and fetches a registry file the
 * first time it is asked for. The same code serves the standalone module and
 * the IMP module build: it wraps the extension's get_data_path, which both
 * shadow modules call.
 *
 * registry.json also names data/academic/ -- licence-gated files the
 * project may not redistribute (FASPR's dun2010bbdep.bin). They are never
 * fetched, explicitly or by an all-fetch, unless BFF_ACADEMIC is set, and
 * no build packages them.
 */
%pythoncode %{
import os as _os

DATA_URL = _os.environ.get("IMP_BFF_DATA_URL", "https://www.peulen.xyz/downloads/imp.bff/")


def _academic_on():
    """BFF_ACADEMIC set to anything but an explicit negative: the user
    asserts the academic-use terms of the opt-in data set (PRD-118)."""
    return _os.environ.get("BFF_ACADEMIC", "").strip().lower() not in (
        "", "0", "false", "no")


def _is_academic(file_name):
    return file_name.replace(_os.sep, "/").startswith("academic/")


def _package_data_dir():
    """The wheel's own IMP/bff/data, or None."""
    d = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
    return d if _os.path.isdir(d) else None


def get_data_cache_dir():
    """Where fetched data files live: IMP_BFF_CACHE, else pooch's per-user
    cache directory for "imp.bff"."""
    d = _os.environ.get("IMP_BFF_CACHE")
    if d:
        return d
    try:
        import pooch
        return str(pooch.os_cache("imp.bff"))
    except ImportError:
        return _os.path.join(_os.path.expanduser("~"), ".cache", "imp.bff")


_installed_get_data_path = _IMP_bff.get_data_path


def get_data_registry():
    """The files fetched on demand: relative path -> "sha256:<hex>"."""
    import json
    candidates = []
    pkg = _package_data_dir()
    if pkg:
        candidates.append(_os.path.join(pkg, "registry.json"))
    try:
        candidates.append(_installed_get_data_path("registry.json"))
    except Exception:
        pass
    for p in candidates:
        try:
            with open(p) as fh:
                return json.load(fh)
        except OSError:
            continue
    return {}


def fetch_data(file_names=None, progressbar=False):
    """Fetch registry files into the cache (all of them when `file_names` is
    None) and return their paths. Needs `pooch`.

    `academic/` files are refused -- and left out of an all-fetch -- unless
    BFF_ACADEMIC is set: their licence lets a user take them for academic
    use, not this package redistribute or auto-download them."""
    registry = get_data_registry()
    if not registry:
        return []
    if file_names is None:
        if not _academic_on():
            file_names = [n for n in registry if not _is_academic(n)]
    else:
        file_names = list(file_names)
    unknown = [n for n in file_names if n not in registry]
    if unknown:
        raise IOException("not in the data registry: " + ", ".join(unknown))
    if not _academic_on():
        gated = [n for n in file_names if _is_academic(n)]
        if gated:
            raise IOException(
                ", ".join(gated) + " is academic-use data (not "
                "redistributable); set BFF_ACADEMIC=1 to download it from "
                + DATA_URL)
    import pooch
    pup = pooch.create(path=get_data_cache_dir(), base_url=DATA_URL, registry=registry)
    return [pup.fetch(n, progressbar=progressbar) for n in file_names]


def _registry_names_under(file_name):
    """The registry files *file_name* stands for: itself, or every file under it
    when it is a directory (C++ asks for ``rotamer_library``, not its files).
    Academic files stay out unless BFF_ACADEMIC is set, as in an all-fetch."""
    registry = get_data_registry()
    name = file_name.replace(_os.sep, "/").rstrip("/")
    if name in registry:
        return [name]
    under = [n for n in registry if n.startswith(name + "/")]
    if not _academic_on():
        under = [n for n in under if not _is_academic(n)]
    return under


def _fetch_into_cache(file_name):
    """Fetch what *file_name* stands for into the cache; False when the registry
    has nothing for it."""
    names = _registry_names_under(file_name)
    if not names:
        return False
    missing = [n for n in names if not _os.path.isfile(_os.path.join(get_data_cache_dir(), n))]
    if missing:
        fetch_data(missing)
    return True


def _fetching_get_data_path(file_name):
    try:
        return _installed_get_data_path(file_name)
    except IOException:
        cached = _os.path.join(get_data_cache_dir(), file_name)
        if _os.path.isfile(cached):
            return cached
        if not _fetch_into_cache(file_name):
            raise
    return _os.path.join(get_data_cache_dir(), file_name)


_fetching_get_data_path.__doc__ = """The path of a data file: as installed, else
from the fetch cache, fetched first when it is one of the registry's
(`fetch_data()` takes them all at once, e.g. before going offline)."""
_IMP_bff.get_data_path = _fetching_get_data_path

# the standalone module reads IMP_BFF_DATA as a PATH-like list: the caller's
# directories, the wheel's own data, the fetch cache -- the cache always, since a
# source build without the large data looks there for what it fetched
_own = _os.environ.get("IMP_BFF_DATA", "")
_os.environ["IMP_BFF_DATA"] = _os.pathsep.join(
    ([_own] if _own else []) + ([_package_data_dir()] if _package_data_dir() else [])
    + [get_data_cache_dir()])

# C++ callers reach the registry through get_data_path()'s fetcher (standalone
# build: the IMP build's get_data_path is IMP's own and reads an installed data/).
if hasattr(_IMP_bff, "set_data_fetcher"):
    class _RegistryFetcher(DataFetcher):
        def fetch(self, file_name):
            try:
                return _fetch_into_cache(file_name)
            except Exception:  # noqa: BLE001 - the C++ caller reports its own miss
                return False

    _registry_fetcher = _RegistryFetcher()   # the C++ side does not own it
    set_data_fetcher(_registry_fetcher)


# IMP's own data (top.lib, par.lib, the element table) where this build links
# IMP and carries it: the C++ side reads IMP_DATA, so a wheel points it at
# what it ships unless the caller has already chosen (PRD-139).
_IMP_DATA = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "imp_data")
if _os.path.isdir(_IMP_DATA) and not _os.environ.get("IMP_DATA"):
    _os.environ["IMP_DATA"] = _IMP_DATA


# --- sequence databases and protein language models --------------------------
# Too large for the registry above: data/sequence_registry.json names them
# (path on DATA_URL, size, sha256, description), and they are fetched only
# when asked for by name -- never by fetch_data()'s all-fetch, never on a
# get_data_path() miss. The download writes `<file>.part`, resumes it with an
# HTTP range request after an interruption, checks the sha256, then renames.


def get_sequence_data_registry():
    """The sequence databases and models: name -> {path, bytes, sha256,
    description, source}."""
    import json
    candidates = []
    pkg = _package_data_dir()
    if pkg:
        candidates.append(_os.path.join(pkg, "sequence_registry.json"))
    try:
        candidates.append(_installed_get_data_path("sequence_registry.json"))
    except Exception:
        pass
    for p in candidates:
        try:
            with open(p) as fh:
                return {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        except OSError:
            continue
    return {}


def get_sequence_data_dir():
    """Where sequence data is fetched to: IMP_BFF_SEQUENCE_DATA, else the
    data cache (get_data_cache_dir()). Databases are tens of GB, so point
    IMP_BFF_SEQUENCE_DATA at a disk with room (ideally an SSD)."""
    return _os.environ.get("IMP_BFF_SEQUENCE_DATA") or get_data_cache_dir()


def _sha256_of(path, block=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_resumable(url, target, size, progressbar=False, block=1 << 22, retries=5):
    """Fetch `url` into `target` + ".part", resuming what is there; returns
    the .part path once it holds `size` bytes."""
    import http.client
    import time
    import urllib.request
    part = target + ".part"
    for attempt in range(retries + 1):
        have = _os.path.getsize(part) if _os.path.exists(part) else 0
        if have == size:
            return part
        if have > size:
            _os.remove(part)
            have = 0
        req = urllib.request.Request(url, headers={"User-Agent": "imp.bff"})
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:      # the server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as out:
                    done, t0, shown = have, time.time(), 0.0
                    while True:
                        chunk = r.read(block)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if progressbar and time.time() - shown > 2:
                            shown = time.time()
                            rate = (done - have) / max(shown - t0, 1e-9) / 1e6
                            print("\r  %s: %.1f / %.1f GB, %.0f MB/s   " % (
                                _os.path.basename(target), done / 1e9, size / 1e9, rate),
                                end="", flush=True)
            if progressbar:
                print()
            if _os.path.getsize(part) == size:
                return part
        except (OSError, http.client.HTTPException) as e:   # URLError, timeouts, resets, short reads
            if attempt == retries:
                raise IOException("download of %s failed: %s" % (url, e))
            time.sleep(min(60, 2 ** attempt))
    raise IOException("download of %s stopped short of %d bytes" % (url, size))


def fetch_sequence_data(name, directory=None, progressbar=False):
    """Fetch a sequence database or a protein language model by name
    (get_sequence_data_registry()) and return its local path.

    A file already there with the right size and checksum is not fetched
    again; an interrupted download resumes. `directory` defaults to
    get_sequence_data_dir().

    >>> o = IMP.bff.ConsurfOptions()
    >>> o.database = IMP.bff.fetch_sequence_data("uniref_consurf")   # 25 GB, once
    """
    registry = get_sequence_data_registry()
    if name not in registry:
        raise IOException("no sequence data named %r; known: %s" % (name, ", ".join(sorted(registry))))
    entry = registry[name]
    if not entry.get("sha256") or entry["sha256"] == "PENDING":
        raise IOException("%s is not published yet (no checksum in the registry)" % name)
    target = _os.path.join(directory or get_sequence_data_dir(), entry["path"])
    _os.makedirs(_os.path.dirname(target), exist_ok=True)
    if _os.path.exists(target) and _os.path.getsize(target) == entry["bytes"]:
        stamp = target + ".sha256"
        if _os.path.exists(stamp) and open(stamp).read().strip() == entry["sha256"]:
            return target                           # verified before; skip re-hashing GBs
        if _sha256_of(target) == entry["sha256"]:
            open(stamp, "w").write(entry["sha256"] + "\n")
            return target
    url = DATA_URL.rstrip("/") + "/" + entry["path"]
    part = _download_resumable(url, target, int(entry["bytes"]), progressbar)
    if progressbar:
        print("  checking sha256 ...", flush=True)
    digest = _sha256_of(part)
    if digest != entry["sha256"]:
        _os.remove(part)
        raise IOException("%s: sha256 %s, the registry says %s; the partial file was removed"
                          % (url, digest, entry["sha256"]))
    _os.replace(part, target)
    open(target + ".sha256", "w").write(digest + "\n")
    return target


def _fetch_data_main(argv=None):
    """`imp_bff_fetch_data`: fetch every registry file, or the ones named;
    with --sequences, sequence databases and protein language models."""
    import argparse
    ap = argparse.ArgumentParser(description=_fetch_data_main.__doc__)
    ap.add_argument("names", nargs="*", help="registry entries; all when none "
                    "(with --sequences: sequence data names, e.g. uniref_consurf)")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--cache", help="fetch into this directory (default: %s)" % get_data_cache_dir())
    ap.add_argument("--sequences", action="store_true",
                    help="sequence databases and protein language models, by name "
                         "(see --list --sequences); fetched to IMP_BFF_SEQUENCE_DATA or --cache")
    a = ap.parse_args(argv)
    if a.cache:
        _os.environ["IMP_BFF_CACHE"] = a.cache
        if a.sequences:
            _os.environ["IMP_BFF_SEQUENCE_DATA"] = a.cache
    if a.sequences:
        registry = get_sequence_data_registry()
        if a.list or not a.names:
            for k, v in sorted(registry.items()):
                state = " (not published yet)" if v.get("sha256") in (None, "", "PENDING") else ""
                print("%-20s %8.2f GB  %s%s" % (k, v["bytes"] / 1e9, v["description"], state))
            if not a.list and not a.names:
                print("\nname what to fetch, e.g.: imp_bff_fetch_data --sequences uniprot_sprot")
            return 0
        for n in a.names:
            print(fetch_sequence_data(n, progressbar=True))
        return 0
    if a.list:
        for k, v in sorted(get_data_registry().items()):
            print(k, v)
        return 0
    for p in fetch_data(a.names or None, progressbar=True):
        print(p)
    return 0
%}
