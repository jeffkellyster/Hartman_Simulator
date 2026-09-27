import json
import pathlib
import subprocess
import sys
import threading
import urllib.request
import zipfile

import numpy as np
import pytest

from hartmann import app

ROOT = pathlib.Path(__file__).resolve().parents[1]


def call(fn, **args):
    response = json.loads(app.handle(json.dumps({"fn": fn, "args": args})))
    assert response["ok"], response.get("error")
    return response["result"]


def fails(fn, **args):
    response = json.loads(app.handle(json.dumps({"fn": fn, "args": args})))
    assert not response["ok"]
    return response["error"]


@pytest.fixture(scope="module")
def lab():
    config = call("lab", noise_sd=0.05, budget=20, seed=3)["config"]
    points = call("design", config=config, kind="maximin_lhs", params={"runs": 10, "seed": 1})["points"]
    measured = call("measure", config=config, X=[], y=[], new=points)
    return config, points, measured["y"]


# --- Setup, designs, measuring ---------------------------------------------------------------


def test_catalog_lists_what_the_page_needs():
    cat = call("catalog")
    assert [f["label"] for f in cat["factors"]][4] == "RF power"
    assert {d["type"] for d in cat["designs"]} >= {"maximin_lhs", "box_behnken", "central_composite"}
    assert {s["method"] for s in cat["strategies"]} == {"bo", "rsm", "random"}
    assert cat["defaults"]["budget"] == 60


def test_lab_fills_in_a_seed_and_describes_the_oracle():
    info = call("lab", noise_sd=0.1, budget=30, scenario=4)
    assert isinstance(info["config"]["seed"], int)
    assert info["config"]["scenario"]["seed"] == 4
    assert info["response"]["header"] == "Thickness non-uniformity (%)"
    assert len(info["bounds"]) == 6
    assert call("lab", units="unit")["response"]["unit"] == ""


def test_designs_come_back_in_range_or_explain_why_not(lab):
    config = lab[0]
    bbd = call("design", config=config, kind="box_behnken")
    assert len(bbd["points"]) == 54
    low, high = np.array(call("oracle_info", config=config)["bounds"]).T
    assert np.all((np.array(bbd["points"]) >= low) & (np.array(bbd["points"]) <= high))
    assert "outside the factor ranges" in fails("design", config=config, kind="central_composite",
                                                params={"alpha": "rotatable"})
    inscribed = call("design", config=config, kind="central_composite", params={"alpha": "rotatable", "inscribed": True})
    assert inscribed["info"]["inscribed"] is True


def test_measuring_counts_earlier_runs_and_repeats_exactly(lab):
    config, points, y = lab
    again = call("measure", config=config, X=[], y=[], new=points)
    assert again["y"] == y
    more = call("measure", config=config, X=points, y=y, new=points[:2])
    assert more["budget"] == {"used": 12, "total": 20, "remaining": 8}
    assert len(more["y_true"]) == 2 and more["y"] != y[:2]  # replicates get fresh noise
    assert "remain" in fails("measure", config=config, X=points, y=y, new=points + points[:1])


def test_tables_round_trip_through_import_and_export(lab):
    config, points, y = lab
    csv = call("export_table", config=config, X=points, y=y)["csv"]
    assert csv.splitlines()[0].startswith("Run,Susceptor temperature (°C)")
    back = call("import_table", config=config, text=csv)
    assert np.allclose(back["points"], points, rtol=1e-5)
    assert np.allclose(back["y"], y, atol=1e-5)
    assert len(back["y_true"]) == len(y)
    planned = call("import_table", config=config, text="RF power (W)\n400\n")
    assert planned["y"] == [None] and len(planned["warnings"]) == 5


# --- Models --------------------------------------------------------------------------------


def test_slices_and_profiles_have_the_right_shapes(lab):
    config, points, y = lab
    s = call("slice2d", config=config, X=points, y=y, i=0, j=4, n=11, truth=True)
    assert len(s["x"]) == len(s["y"]) == 11
    assert np.array(s["model"]).shape == (11, 11) and np.array(s["true"]).shape == (11, 11)
    assert s["x"][0] == 250 and s["y"][-1] == 800
    p = call("profile", config=config, X=points, y=y, n=9, truth=True)
    assert len(p["traces"]) == 6 and len(p["traces"][2]["mean"]) == 9
    assert set(p["at"]) == {"mean", "sd", "true"}
    sd = call("slice2d", config=config, X=points, y=y, i=1, j=2, n=5, layer="sd")
    assert np.all(np.array(sd["model"]) >= 0)
    ei = call("slice2d", config=config, X=points, y=y, i=1, j=2, n=5, layer="acquisition")
    assert np.all(np.array(ei["model"]) >= 0)
    truth_only = call("slice2d", config=config, X=[], y=[], i=0, j=1, n=5, model=None, truth=True)
    assert "model" not in truth_only and "true" in truth_only


def test_model_calls_explain_bad_requests(lab):
    config, points, y = lab
    assert "two different factors" in fails("slice2d", config=config, X=points, y=y, i=2, j=2)
    assert "grid size" in fails("slice2d", config=config, X=points, y=y, i=0, j=1, n=500)
    assert "at least 2" in fails("fit", config=config, X=points[:1], y=y[:1])
    assert "unknown function" in fails("launch_rockets")


def test_fit_describes_the_gp_and_the_rsm(lab):
    config, points, y = lab
    gp = call("fit", config=config, X=points, y=y)
    assert gp["type"] == "gp" and set(gp["lengthscales"]) == {f["label"] for f in call("catalog")["factors"]}
    rsm = call("fit", config=config, X=points, y=y, model="rsm")
    assert rsm["type"] == "rsm" and rsm["underdetermined"] and rsm["terms"] == 28


def test_suggestions_repeat_and_respect_the_budget(lab):
    config, points, y = lab
    a = call("suggest", config=config, X=points, y=y, q=3, seed=5)
    b = call("suggest", config=config, X=points, y=y, q=3, seed=5)
    assert a["points"] == b["points"] and len(a["points"]) == 3
    tight = {**config, "budget": 11}
    assert len(call("suggest", config=tight, X=points, y=y, q=3)["points"]) == 1
    assert "used up" in fails("suggest", config={**config, "budget": 10}, X=points, y=y)
    rsm = call("suggest", config=config, X=points, y=y, method="rsm")
    assert "not unique" in rsm["warnings"][0]
    assert len(call("suggest", config=config, X=points, y=y, method="random", q=2)["points"]) == 2


# --- Benchmark ------------------------------------------------------------------------------


def test_live_benchmark_runs_match_the_kernel_and_summarize():
    config = call("lab", noise_sd=0.05, seed=100)["config"]
    run = call("benchmark_run", config=config, strategy={"method": "random"}, replicate=1, budget=8)
    assert run["label"] == "Random" and len(run["observed"]) == len(run["true"]) == 8
    assert all(np.diff(run["observed"]) <= 0)
    curves = {"Random": {"observed": [run["observed"]] * 3, "true": [run["true"]] * 3}}
    summary = call("benchmark_summary", curves=curves, optimum=1.016)
    assert summary["replicates"] == 3 and summary["budget"] == 8
    assert summary["summary"]["true"]["Random"]["median"] == pytest.approx(run["true"])


def test_the_saved_comparison_matches_what_the_page_expects():
    data = json.loads((ROOT / "web" / "data" / "benchmark.json").read_text(encoding="utf-8"))
    assert data["labels"] == ["Random", "LHS + RSM", "BO (EI)", "BO (UCB)"]
    for which in ("true", "observed"):
        for label in data["labels"]:
            stats = data["summary"][which][label]
            assert len(stats["median"]) == data["budget"]
            assert all(lo <= m <= hi for lo, m, hi in zip(stats["q25"], stats["median"], stats["q75"]))
    assert data["summary"]["true"]["BO (UCB)"]["median"][-1] < data["summary"]["true"]["Random"]["median"][-1]


# --- Packaging for the browser ----------------------------------------------------------------


def test_the_engine_loads_in_the_browser_without_scipy():
    code = "import sys, hartmann.app; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_engine_zip_holds_both_packages():
    sys.path.insert(0, str(ROOT / "scripts"))
    import serve

    with zipfile.ZipFile(serve.build_engine_zip()) as archive:
        names = set(archive.namelist())
    assert {"hartmann/app.py", "hartmann/__init__.py", "seqopt/gp.py", "seqopt/__init__.py"} <= names
    assert not any("__pycache__" in n for n in names)


def test_native_mode_serves_the_api_and_its_own_worker():
    sys.path.insert(0, str(ROOT / "scripts"))
    import functools
    import http.server

    import serve

    handler = functools.partial(serve.NativeHandler, directory=str(ROOT / "web"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        body = json.dumps({"fn": "catalog", "args": {}}).encode()
        with urllib.request.urlopen(urllib.request.Request(f"{base}/api", data=body)) as r:
            assert json.loads(r.read())["ok"] is True
        with urllib.request.urlopen(f"{base}/worker.js") as r:
            assert b"/api" in r.read()
        with urllib.request.urlopen(f"{base}/index.html") as r:
            assert b"Hartmann 6D Optimization Simulator" in r.read()
    finally:
        server.shutdown()
