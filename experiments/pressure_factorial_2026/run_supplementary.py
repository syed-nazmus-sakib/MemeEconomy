#!/usr/bin/env python3
"""
Run the two supplementary contrast arms (reward_only, threat_only) for the
pressure-factor rebuttal experiment. These are descriptive arms excluded from
the confirmatory 2x2x2 factorial (like the neutral arm):

  reward_only : intensified objective, NO target number, NO elimination,
                NO survival/threat vocabulary  (DPD4-Q1 control)
  threat_only : existential menace, NO karma target, NO threshold, NO gap
                math  (falsification cell for the "adversarial prompting"
                alternative raised by Reviewer vpDE)

Uses the same 33 instances, same rep-1 ordering, and same run seed as the
preregistered factorial so paired contrasts against any factorial cell are
valid. Skips cells whose result dir already holds a completed run.

Usage: python3 run_supplementary.py [--jobs N] [--repeats 1 ...]
"""
import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiment import (  # noqa: E402
    REPO, load_manifest, ordered_instances, run_is_complete,
)

CONDITIONS = ["reward_only", "threat_only"]


def build_supplementary_tasks(manifest, repeats):
    tasks = []
    for rep in repeats:
        seed = manifest["replicate_seeds"][rep - 1]
        instances = ordered_instances(manifest, seed)
        for cond in CONDITIONS:
            for model in manifest["models"]:
                label = f"pressure_factorial_2026/{cond}/rep_{rep:02d}"
                result_dir = REPO / "results" / label / model
                cmd = [
                    sys.executable, "-m", "memeeconomy.run_single_agent",
                    "--models", model,
                    "--rounds", str(len(instances)),
                    "--condition", cond,
                    "--arch", "bdi",
                    "--seed", str(seed),
                    "--run-label", f"_pf_{cond}_r{rep:02d}",
                    "--results-label", label,
                    "--instances", *instances,
                ]
                tasks.append({
                    "model": model, "condition": cond, "replicate": rep,
                    "seed": seed, "instances": instances,
                    "result_dir": str(result_dir), "command": cmd,
                })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--repeats", nargs="+", type=int, default=[1])
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest()
    tasks = build_supplementary_tasks(manifest, args.repeats)
    pending = tasks if args.rerun else [t for t in tasks if not run_is_complete(t)]
    print(f"[supplementary] {len(tasks)} cells; {len(pending)} pending; "
          f"{sum(len(t['instances']) for t in pending)} decisions", flush=True)

    def run(task):
        print(f"[supplementary] START {task['condition']}/{task['model']}"
              f"/rep{task['replicate']}", flush=True)
        r = subprocess.run(task["command"], cwd=str(REPO), check=False)
        print(f"[supplementary] DONE  {task['condition']}/{task['model']}"
              f"/rep{task['replicate']} rc={r.returncode}", flush=True)
        return r.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        codes = list(pool.map(run, pending))
    print(f"[supplementary] complete: {codes.count(0)}/{len(codes)} cells rc=0", flush=True)


if __name__ == "__main__":
    main()
