"""Package the JMP add-in.

    python scripts/build_addin.py

Zips jmp/addin/ (including the engine wheel, if build_release.py put one
there) plus a copy of jmp/README.md into jmp/HartmannSimulator.jmpaddin.
Install it by opening the file in JMP (File > Open) or dragging it onto the
JMP Home window.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDIN = ROOT / "jmp" / "addin"
TARGET = ROOT / "jmp" / "HartmannSimulator.jmpaddin"


def main():
    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ADDIN.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                archive.write(path, path.name)
        archive.write(ROOT / "jmp" / "README.md", "README.md")
    names = zipfile.ZipFile(TARGET).namelist()
    print(f"wrote {TARGET.relative_to(ROOT)} with {len(names)} files: {', '.join(names)}")


if __name__ == "__main__":
    main()
