import ast
import pathlib
import subprocess
import sys

SEQOPT = pathlib.Path(__file__).resolve().parents[1] / "seqopt"


def test_seqopt_source_never_imports_hartmann():
    for path in SEQOPT.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "hartmann" for n in names), f"{path.name} imports {names}"


def test_importing_seqopt_does_not_load_hartmann():
    out = subprocess.run(
        [sys.executable, "-c", "import sys, seqopt; print('hartmann' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"
