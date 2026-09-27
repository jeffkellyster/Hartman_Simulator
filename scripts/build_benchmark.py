"""Build the saved strategy comparison the browser app opens with (web/data/benchmark.json).

    OPENBLAS_NUM_THREADS=1 python scripts/build_benchmark.py [--replicates 20] [--budget 60]

Runs random search, LHS + RSM, BO with expected improvement and BO with a
confidence bound on the default lab (PECVD process units, noise SD 0.05 %),
and keeps the median and quartiles of both curves. About four minutes on a
laptop. Rerun it after changing the optimization kernel.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hartmann import HartmannOracle, NoiseModel  # noqa: E402
from seqopt.benchmark import run_benchmark  # noqa: E402
from seqopt.loop import Strategy  # noqa: E402

TARGET = ROOT / "web" / "data" / "benchmark.json"
STRATEGIES = [Strategy("random"), Strategy("rsm"), Strategy("bo", acquisition="ei"), Strategy("bo", acquisition="ucb")]


def rounded(value, digits=5):
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, list):
        return [rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--noise-sd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    base = HartmannOracle(noise=NoiseModel(sd=args.noise_sd), seed=args.seed)
    config = {**base.config, "budget": None}

    def progress(done, total, label):
        print(f"\r{done}/{total} runs ({label})".ljust(50), end="", file=sys.stderr, flush=True)

    result = run_benchmark(lambda r: HartmannOracle.from_config({**config, "seed": args.seed + r}),
                           STRATEGIES, args.budget, args.replicates, seed=0, optimum=base.optimum()["y"],
                           progress=progress)
    print(file=sys.stderr)
    data = result.to_dict()
    data.pop("curves")  # the page only draws the summary
    data["oracle"] = config
    data["generated"] = datetime.date.today().isoformat()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(rounded(data), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size // 1024} KB)")
    for label, stats in result.at(args.budget).items():
        print(f"  {label:12s} median {stats['median']:.3f} [{stats['q25']:.3f}, {stats['q75']:.3f}]")


if __name__ == "__main__":
    main()
