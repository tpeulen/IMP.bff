"""Source layout: every header belongs to exactly one theme, every source to a theme.

IMP links only include/*.h and include/internal/*.h into its build tree, so
headers stay flat and src/<theme>/Headers.cmake assigns them. Sources live in
src/<theme>/ (one level: the depth IMP's setup_cmake globs).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LAYER = "imp"
# themes whose sources may not include the headers of these themes (layering)
FORBIDDEN = {
    "util": {"probe", "structure", LAYER},
    "graph": {"probe", "structure", LAYER},
}
# Known upward includes. A shrinking list, never somewhere to add yourself:
# GraphNodeRegistry is the one factory that registers every node type, so it
# names the fret/spectroscopy nodes (allowed above); Registry.cpp still names
# the probe data paths.
KNOWN = {("util/Registry.cpp", "ProbeDataPaths.h")}


def _list(path, var):
    m = re.search(r'set\(%s\s+"([^"]*)"\)' % re.escape(var), path.read_text(), re.S)
    assert m, (path, var)
    return [x for x in m.group(1).replace("\n", "").split(";") if x]


def _themes():
    return _list(SRC / "Themes.cmake", "imp_bff_themes")


def _public(theme):
    return _list(SRC / theme / "Headers.cmake", f"imp_bff_{theme}_headers")


def _private(theme):
    return _list(SRC / theme / "Headers.cmake", f"imp_bff_{theme}_private_sources")


def _layer_headers():
    return [h for h in _list(SRC / LAYER / "Headers.cmake", "imp_bff_layer_headers")
            if "/" not in h]


def test_every_header_has_exactly_one_theme():
    owners = {}
    for theme in _themes():
        for h in _public(theme):
            assert h not in owners, f"{h} in {owners[h]} and {theme}"
            owners[h] = theme
    for h in _layer_headers():
        assert h not in owners, f"{h} in {owners[h]} and {LAYER}"
        owners[h] = LAYER
    flat = {p.name for p in (ROOT / "include").glob("*.h")} - {"bff_config.h"}
    assert flat == set(owners), (sorted(flat - set(owners)), sorted(set(owners) - flat))


def test_every_source_sits_in_its_theme():
    for theme in _themes():
        public = {h[:-2] for h in _public(theme)}
        private = {s[:-4] for s in _private(theme)}
        for path in sorted((SRC / theme).glob("*.cpp")):
            assert path.stem in public | private, f"{path.relative_to(ROOT)} is not named in {theme}/Headers.cmake"
        for stem in private:
            assert (SRC / theme / f"{stem}.cpp").is_file(), (theme, stem)
    flat = sorted(p.name for p in SRC.glob("*.cpp"))
    assert flat == ["ImpLayer.cpp", "ThemeSources.cpp"], flat
    included = re.findall(r'#include "([a-z]+/[A-Za-z0-9_]+\.cpp)"', (SRC / "ThemeSources.cpp").read_text())
    on_disk = sorted(f"{t}/{p.name}" for t in _themes() for p in (SRC / t).glob("*.cpp"))
    assert sorted(included) == on_disk, set(included) ^ set(on_disk)


def test_themes_do_not_reach_upward():
    owners = {}
    for theme in _themes():
        for h in _public(theme):
            owners[h] = theme
    for h in _layer_headers():
        owners[h] = LAYER
    for theme, banned in FORBIDDEN.items():
        for path in sorted((SRC / theme).glob("*.cpp")):
            for inc in re.findall(r'#include\s*[<"]IMP/bff/([A-Za-z0-9_]+\.h)[>"]', path.read_text()):
                if (f"{theme}/{path.name}", inc) in KNOWN:
                    continue
                assert owners.get(inc) not in banned, f"{path.relative_to(ROOT)} includes {inc} ({owners[inc]})"
