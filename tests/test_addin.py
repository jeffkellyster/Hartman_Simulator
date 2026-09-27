"""Checks on the JMP add-in that can run without JMP.

JSL itself only runs inside JMP, so these catch what can be caught here:
unbalanced brackets, Python blocks that don't compile or call adapter
functions that don't exist, menu entries pointing at missing files, and a
committed .jmpaddin (with its bundled engine) that no longer matches the
source.
"""

import ast
import pathlib
import re
import tomllib
import zipfile

import pytest

from hartmann import jmp_adapter

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDIN = ROOT / "jmp" / "addin"
PACKAGE = ROOT / "jmp" / "HartmannSimulator.jmpaddin"
SCRIPTS = sorted(ADDIN.glob("*.jsl"))
ADDIN_ID = dict(line.split("=", 1) for line in (ADDIN / "addin.def").read_text().splitlines())["id"]
REBUILD = "run `python scripts/build_release.py` to rebuild the add-in"


def strip_jsl(text: str) -> tuple[str, list[str]]:
    """JSL with strings and comments blanked out, and the raw-string blocks ("\\[ ... ]\\") it held."""
    out, raws, i = [], [], 0
    while i < len(text):
        if text.startswith('"\\[', i):
            end = text.index(']\\"', i + 3)
            raws.append(text[i + 3:end])
            out.append('""')
            i = end + 3
        elif text[i] == '"':
            i += 1
            while text[i] != '"':
                i += 3 if text.startswith("\\!", i) else 1
            out.append('""')
            i += 1
        elif text.startswith("//", i):
            i = text.index("\n", i) if "\n" in text[i:] else len(text)
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out), raws


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_jsl_brackets_balance(path):
    code, _ = strip_jsl(path.read_text(encoding="utf-8"))
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for line_no, line in enumerate(code.splitlines(), 1):
        for ch in line:
            if ch in "([{":
                stack.append((ch, line_no))
            elif ch in pairs:
                assert stack and stack[-1][0] == pairs[ch], f"{path.name}:{line_no}: unexpected {ch!r}"
                stack.pop()
    assert not stack, f"{path.name}: unclosed {stack[-1][0]!r} from line {stack[-1][1]}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_each_script_starts_in_its_own_namespace_and_uses_this_addin(path):
    text = path.read_text(encoding="utf-8")
    code, _ = strip_jsl(text)
    assert code.lstrip().startswith("Names Default To Here( 1 );"), "the first statement must be Names Default To Here( 1 )"
    for addin_id in re.findall(r"\$ADDIN_HOME\(([^)]*)\)", text):
        assert addin_id == ADDIN_ID
    for target in re.findall(r'Include\( "\$ADDIN_HOME\([^)]*\)/([^"]+)" \)', text):
        assert (ADDIN / target).exists(), f"{path.name} includes missing {target}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_embedded_python_compiles_and_calls_real_adapter_functions(path):
    _, raws = strip_jsl(path.read_text(encoding="utf-8"))
    for block in raws:
        tree = ast.parse(block)  # raises SyntaxError with the line if a block is broken
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "jmp_adapter":
                assert hasattr(jmp_adapter, node.attr), f"{path.name} calls jmp_adapter.{node.attr}, which doesn't exist"


def test_every_menu_item_points_at_a_script():
    menu = (ADDIN / "addin.jmpcust").read_text(encoding="utf-8")
    actions = re.findall(r"\$ADDIN_HOME\(([^)]*)\)\\([\w.]+)</jm:action>", menu)
    assert len(actions) == 7
    for addin_id, script in actions:
        assert addin_id == ADDIN_ID and (ADDIN / script).exists(), script


def test_the_committed_addin_matches_the_source():
    with zipfile.ZipFile(PACKAGE) as archive:
        names = set(archive.namelist())
        for path in ADDIN.iterdir():
            if path.is_file() and path.suffix != ".whl" and not path.name.startswith("."):
                assert path.name in names, f"{path.name} is missing from the add-in; {REBUILD}"
                assert archive.read(path.name) == path.read_bytes(), f"{path.name} changed; {REBUILD}"
        assert archive.read("README.md") == (ROOT / "jmp" / "README.md").read_bytes(), f"README.md changed; {REBUILD}"
        wheels = [n for n in names if n.endswith(".whl")]
        assert len(wheels) == 1, f"expected one bundled engine wheel; {REBUILD}"
        wheel_bytes = archive.read(wheels[0])
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert wheels[0] == f"hartmann_sim-{version}-py3-none-any.whl", REBUILD

    import io

    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        bundled = {n: wheel.read(n) for n in wheel.namelist() if n.endswith(".py")}
    source = {p.relative_to(ROOT).as_posix(): p.read_bytes()
              for package in ("hartmann", "seqopt") for p in (ROOT / package).glob("*.py")}
    assert set(bundled) == set(source), f"the bundled engine has different modules; {REBUILD}"
    stale = sorted(n for n in source if bundled[n] != source[n])
    assert not stale, f"the bundled engine is out of date ({', '.join(stale)}); {REBUILD}"
