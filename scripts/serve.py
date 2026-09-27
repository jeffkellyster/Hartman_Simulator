"""Serve the browser app locally.

    python scripts/serve.py [--port 8000] [--native]

Packages the engine into web/engine.zip (the page loads it into Pyodide),
then serves web/ at http://localhost:8000/. Run it again after changing the
engine so the page picks up the new code.

--native runs the engine in this Python process instead of in the browser:
the page is the same, but its worker forwards each call to /api here. It
needs no internet (Pyodide comes from a CDN otherwise), starts instantly and
runs at native speed, which makes it the quickest way to develop and test.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import threading
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
PACKAGES = ("hartmann", "seqopt")
NATIVE_WORKER = Path(__file__).resolve().parent / "worker_native.js"


def build_engine_zip() -> Path:
    target = WEB / "engine.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for package in PACKAGES:
            for path in sorted((ROOT / package).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT).as_posix())
    return target


class NativeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files, plus /api (the engine's JSON dispatcher) and a worker that calls it."""

    lock = threading.Lock()

    def do_GET(self):
        if self.path.split("?")[0] == "/worker.js":
            body = NATIVE_WORKER.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api":
            self.send_error(404)
            return
        from hartmann import app

        request = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
        with self.lock:  # the dispatcher keeps a model cache; one call at a time
            body = app.handle(request).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        if len(args) > 1 and str(args[1]).isdigit() and int(args[1]) >= 400:
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--native", action="store_true", help="run the engine here instead of in the browser")
    args = parser.parse_args()

    zip_path = build_engine_zip()
    print(f"packaged engine -> {zip_path.relative_to(ROOT)}")
    if args.native:
        # The engine's matrices are small: extra BLAS threads only spin, and they fight the
        # browser for the CPU. Set before NumPy is first imported (on the first /api call).
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        sys.path.insert(0, str(ROOT))
        handler = functools.partial(NativeHandler, directory=str(WEB))
    else:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        mode = " (native engine)" if args.native else ""
        print(f"serving http://localhost:{args.port}/{mode}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
