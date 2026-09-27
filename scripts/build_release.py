"""Build everything a colleague needs.

    python scripts/build_release.py

1. The engine wheel (dist/hartmann_sim-<version>-py3-none-any.whl).
2. The JMP add-in with the wheel bundled (jmp/HartmannSimulator.jmpaddin, committed
   so colleagues can download it straight from GitHub).
3. The static browser app (site/).
4. One zip to hand out: dist/HartmannSimulator-<version>.zip, with the add-in,
   the wheel, the browser app and a short README.

Run it after changing the engine or jmp/addin/; a test fails when the committed
add-in no longer matches the source.
"""

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

COLLEAGUE_README = """Hartmann 6D Optimization Simulator

JMP (18 or 19):
  1. Open HartmannSimulator.jmpaddin in JMP (File > Open, or drag it onto the Home window).
  2. Add-Ins > Hartmann Simulator > Install or Update Engine (once; it also installs NumPy and SciPy).
  3. Add-Ins > Hartmann Simulator > Set Up a Design... and go from there.
     The Help item in that menu explains the rest.

Browser app (no install):
  Open the hosted link if you were given one. To serve it yourself you need Python 3:
      python -m http.server 8000 --directory site
  then open http://localhost:8000/ (it must be served over HTTP; opening
  index.html directly won't work). The page downloads Pyodide from its CDN.

Python (3.11+):
  pip install hartmann_sim-*.whl
  hartmann info
  >>> from hartmann import HartmannOracle
"""


def main():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob("*.whl"):
        old.unlink()
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--wheel-dir", str(DIST), "-q"],
                   check=True)
    wheel = next(DIST.glob("hartmann_sim-*.whl"))
    addin_dir = ROOT / "jmp" / "addin"
    for old in addin_dir.glob("*.whl"):
        old.unlink()
    shutil.copy(wheel, addin_dir / wheel.name)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_addin.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")], check=True)

    target = DIST / f"HartmannSimulator-{version}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", COLLEAGUE_README)
        archive.write(ROOT / "jmp" / "HartmannSimulator.jmpaddin", "HartmannSimulator.jmpaddin")
        archive.write(wheel, wheel.name)
        archive.write(ROOT / "jmp" / "README.md", "JMP-guide.md")
        for path in sorted((ROOT / "site").rglob("*")):
            if path.is_file():
                archive.write(path, f"site/{path.relative_to(ROOT / 'site').as_posix()}")
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
