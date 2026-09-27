"""Build the static browser app into site/ (what GitHub Pages serves).

    python scripts/build_site.py

Copies web/ (including the saved comparison in web/data/) and packages the
engine into site/engine.zip. The result works from any static host over
HTTP; Pyodide loads from its CDN.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve import ROOT, WEB, build_engine_zip  # noqa: E402

SITE = ROOT / "site"


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(WEB, SITE, ignore=shutil.ignore_patterns("engine.zip", ".*"))
    build_engine_zip()
    shutil.copy(WEB / "engine.zip", SITE / "engine.zip")
    print(f"built {SITE.relative_to(ROOT)}: {', '.join(sorted(p.name for p in SITE.iterdir()))}")


if __name__ == "__main__":
    main()
