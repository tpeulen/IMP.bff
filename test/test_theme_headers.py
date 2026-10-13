"""Source layout: headers and sources sit in their theme's directory.

Public headers live in include/<theme>/ and sources in src/<theme>/ (one level
each; imp/ is the connection layer to IMP). The directory is the assignment,
so the checks here are that nothing is left flat, that a source sits in the
theme of its header, that the generated files (tools/dev/theme_layout.py) are
current, that every include names the themed path, and that the lower themes
do not reach upward.
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INCLUDE = ROOT / "include"
LAYER = "imp"
# themes whose sources and headers may not include the headers of these themes
FORBIDDEN = {
    "util": {"probe", "structure", LAYER},
    "graph": {"probe", "structure", LAYER},
}
# Known upward includes. A shrinking list, never somewhere to add yourself:
# Registry.cpp still names the probe data paths.
KNOWN = {("src/util/Registry.cpp", "probe/ProbeDataPaths.h")}
_INCLUDE = re.compile(r'(?:#|%)include\s*[<"]IMP/bff/([^>"]+\.h)[>"]')


def _themes():
    m = re.search(r'set\(imp_bff_themes\s+"([^"]*)"\)', (SRC / "Themes.cmake").read_text())
    return [t for t in m.group(1).split(";") if t]


def _layout():
    spec = importlib.util.spec_from_file_location("theme_layout", ROOT / "tools/dev/theme_layout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _owners():
    """{header file name: theme}"""
    owners = {}
    for theme in _themes() + [LAYER]:
        for h in sorted((INCLUDE / theme).glob("*.h")):
            assert h.name not in owners, f"{h.name} in {owners[h.name]} and {theme}"
            owners[h.name] = theme
    return owners


def test_include_holds_only_themes_and_umbrellas():
    dirs = {p.name for p in INCLUDE.iterdir() if p.is_dir()}
    assert dirs == set(_themes()) | {LAYER, "internal"}, dirs
    flat = {p.name for p in INCLUDE.glob("*.h")}
    assert flat == {f"{t}.h" for t in _themes() + [LAYER]}, flat
    assert _owners()


def test_every_source_sits_in_its_theme():
    owners = _owners()
    for theme in _themes() + [LAYER]:
        for path in sorted((SRC / theme).glob("*.cpp")):
            header = path.stem + ".h"
            if header in owners:
                assert owners[header] == theme, (
                    f"{path.relative_to(ROOT)} implements include/{owners[header]}/{header}")
    flat = sorted(p.name for p in SRC.glob("*.cpp"))
    assert flat == ["ImpLayer.cpp", "ThemeSources.cpp"], flat


def test_generated_layout_files_are_current():
    layout = _layout()
    import os
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        expected = layout.outputs()
    finally:
        os.chdir(cwd)
    for rel, content in expected.items():
        assert (ROOT / rel).read_text() == content, (
            f"{rel} is stale: run tools/dev/theme_layout.py")


def _include_sites():
    for d in ("include", "src", "pyext", "bin", "test", "examples", "standalone"):
        for path in sorted((ROOT / d).rglob("*")):
            if path.suffix in {".h", ".cpp", ".i", ".i-in", ".py"} and path.is_file():
                yield path


def test_every_include_names_the_themed_path():
    owners = _owners()
    for path in _include_sites():
        for inc in _INCLUDE.findall(path.read_text(encoding="utf-8", errors="replace")):
            name = inc.rsplit("/", 1)[-1]
            if name in owners:
                assert inc == f"{owners[name]}/{name}", (
                    f"{path.relative_to(ROOT)} includes IMP/bff/{inc}, "
                    f"not IMP/bff/{owners[name]}/{name}")


def test_themes_do_not_reach_upward():
    owners = _owners()
    for theme, banned in FORBIDDEN.items():
        for path in sorted((SRC / theme).glob("*.cpp")) + sorted((INCLUDE / theme).glob("*.h")):
            rel = str(path.relative_to(ROOT))
            for inc in _INCLUDE.findall(path.read_text()):
                if (rel, inc) in KNOWN:
                    continue
                owner = owners.get(inc.rsplit("/", 1)[-1])
                assert owner not in banned, f"{rel} includes {inc} ({owner})"
