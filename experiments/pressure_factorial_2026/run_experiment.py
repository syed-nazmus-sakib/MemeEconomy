#!/usr/bin/env python3
"""Build, validate, and optionally execute the matched pressure experiment."""

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST_PATH = HERE / "manifest.json"
RESULTS_ROOT = REPO / "results" / "pressure_factorial_2026"


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ordered_instances(manifest, seed, limit=None):
    items = list(manifest["instances"])
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def selected_values(requested, available, label):
    if not requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}")
    return [x for x in available if x in requested]


def build_tasks(manifest, models=None, conditions=None, repeats=None, smoke=False):
    all_conditions = manifest["core_conditions"] + manifest["reference_conditions"]
    models = selected_values(models, manifest["models"], "models")
    conditions = selected_values(conditions, all_conditions, "conditions")
    rep_ids = list(range(1, len(manifest["replicate_seeds"]) + 1))
    if repeats:
        bad = sorted(set(repeats) - set(rep_ids))
        if bad:
            raise ValueError(f"Unknown repetitions: {bad}")
        rep_ids = [r for r in rep_ids if r in repeats]

    tasks = []
    for rep in rep_ids:
        seed = manifest["replicate_seeds"][rep - 1]
        instances = ordered_instances(manifest, seed, limit=3 if smoke else None)
        for condition in conditions:
            for model in models:
                result_rel = Path("pressure_factorial_2026") / condition / f"rep_{rep:02d}"
                run_label = f"_pf_{condition}_r{rep:02d}"
                cmd = [
                    sys.executable, "-m", "memeeconomy.run_single_agent",
                    "--models", model,
                    "--rounds", str(len(instances)),
                    "--condition", condition,
                    "--arch", "bdi",
                    "--seed", str(seed),
                    "--run-label", run_label,
                    "--results-label", str(result_rel),
                    "--instances", *instances,
                ]
                tasks.append({
                    "model": model,
                    "condition": condition,
                    "replicate": rep,
                    "seed": seed,
                    "instances": instances,
                    "result_dir": str(REPO / "results" / result_rel / model),
                    "command": cmd,
                })
    # Randomize wall-clock ordering independently of outcomes/API provider.
    random.Random(35304).shuffle(tasks)
    return tasks


def run_is_complete(task):
    path = Path(task["result_dir"])
    if not path.exists():
        return False
    for log in sorted(path.glob("run*.jsonl"), reverse=True):
        try:
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]
        except Exception:
            continue
        starts = [r for r in rows if r.get("type") == "run_start"]
        ends = [r for r in rows if r.get("type") == "run_end"]
        if (starts and ends and starts[0].get("condition") == task["condition"]
                and starts[0].get("run_seed") == task["seed"]):
            return True
    return False


def validate_manifest(manifest):
    sys.path.insert(0, str(REPO))
    from memeeconomy.prompts import PRESSURE_FACTORIAL_CONDITIONS

    expected = {
        (target, elimination, aggressive)
        for target in (25_000, 50_000)
        for elimination in (False, True)
        for aggressive in (False, True)
    }
    actual = {
        (
            spec["target_karma"],
            spec["elimination_threshold"] is not None,
            bool(spec["aggressive"]),
        )
        for name, spec in PRESSURE_FACTORIAL_CONDITIONS.items()
        if name in manifest["core_conditions"]
    }
    if actual != expected or len(manifest["core_conditions"]) != 8:
        raise RuntimeError("Core conditions do not form a complete 2 x 2 x 2 design")

    cfg = json.loads((REPO / "configs" / "instance_config.json").read_text(encoding="utf-8"))
    task_cfg = json.loads((REPO / "memeeconomy_tasks.json").read_text(encoding="utf-8"))["injected_tasks"]
    if len(manifest["instances"]) != 33 or len(set(manifest["instances"])) != 33:
        raise RuntimeError("Manifest must contain 33 unique instances")
    for name in manifest["instances"]:
        tiers = {m["tier"] for m in cfg["instances"][name]["memes"].values()}
        if not ({1, 5} <= tiers) or task_cfg[name].get("url", "#") == "#":
            raise RuntimeError(f"Ineligible instance in manifest: {name}")

    eligible = []
    for name, instance in cfg["instances"].items():
        tiers = {m["tier"] for m in instance["memes"].values()}
        if {1, 5} <= tiers and task_cfg.get(name, {}).get("url", "#") != "#":
            eligible.append(name)
    salt = manifest["instance_selection"]["salt"]
    expected_instances = sorted(
        eligible,
        key=lambda name: hashlib.sha256(f"{salt}\0{name}".encode("utf-8")).hexdigest(),
    )[:manifest["instance_selection"]["n"]]
    if manifest["instances"] != expected_instances:
        raise RuntimeError("Frozen instance list does not match the preregistered hash selection")


def preflight():
    failures = []
    if not os.environ.get("OPENROUTER_KEY"):
        failures.append("OPENROUTER_KEY is unset")
    for module in ("playwright", "jinja2"):
        if importlib.util.find_spec(module) is None:
            failures.append(f"Python package missing: {module}")
    try:
        with urllib.request.urlopen("http://localhost:9999", timeout=3) as response:
            if response.status >= 500:
                failures.append(f"benchmark server returned HTTP {response.status}")
    except Exception as exc:
        failures.append(f"benchmark server unavailable on :9999 ({type(exc).__name__})")
    docker = subprocess.run(
        ["docker", "inspect", "reddit_benchmark"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if docker.returncode != 0:
        failures.append("Docker container reddit_benchmark is unavailable")
    return failures


def execute_one(task):
    label = f"{task['model']} | {task['condition']} | rep {task['replicate']}"
    print(f"[START] {label}", flush=True)
    result = subprocess.run(task["command"], cwd=str(REPO), check=False)
    print(f"[{'DONE' if result.returncode == 0 else 'FAIL'}] {label}", flush=True)
    return label, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute paid API runs; default is dry-run")
    parser.add_argument("--smoke", action="store_true", help="Use three instances and repetition 1")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent benchmark processes")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int)
    parser.add_argument("--rerun", action="store_true", help="Do not skip completed cells")
    args = parser.parse_args()

    manifest = load_manifest()
    validate_manifest(manifest)
    repeats = [1] if args.smoke else args.repeats
    tasks = build_tasks(
        manifest,
        models=args.models,
        conditions=args.conditions,
        repeats=repeats,
        smoke=args.smoke,
    )
    pending = tasks if args.rerun else [t for t in tasks if not run_is_complete(t)]
    plan = {
        "experiment_id": manifest["experiment_id"],
        "smoke": args.smoke,
        "total_cells": len(tasks),
        "pending_cells": len(pending),
        "planned_decisions": sum(len(t["instances"]) for t in pending),
        "tasks": tasks,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "execution_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Validated {len(tasks)} cells; {len(pending)} pending; "
          f"{plan['planned_decisions']} planned decisions.")
    print(f"Execution plan: {HERE / 'execution_plan.json'}")

    if not args.execute:
        print("Dry-run only. Add --execute after preflight is green.")
        return 0

    failures = preflight()
    if failures:
        print("Preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    if not pending:
        print("All selected cells are already complete.")
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(execute_one, pending))
    failed = [label for label, code in results if code != 0]
    if failed:
        print(f"Failed cells ({len(failed)}):", file=sys.stderr)
        for label in failed:
            print(f"  - {label}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
