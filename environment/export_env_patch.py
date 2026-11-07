#!/usr/bin/env python3
"""
export_env_patch.py
===================
Maintainer tool. Exports the MemeEconomy rows of a running environment — megathread
submissions, their comments, comment authors and forums, with their original ids,
usernames and timestamps — so that `setup_environment.py --exact` can restore the
exact environment the paper's agents saw onto a stock Reddit WebArena container.

Output (published in the Hugging Face dataset under env_patch/):
    users.csv.gz, forums.csv.gz, submissions.csv.gz, comments.csv.gz
    tasks.json        instance -> megathread (the injected_tasks map)
    manifest.json     row counts and sha256 per file

Usage:
    python environment/export_env_patch.py --split paper98 --out env_patch
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "instance_config.json"
TASKS_PATH  = ROOT / "memeeconomy_tasks.json"

# Restore order: every table only references tables earlier in the list.
TABLE_QUERIES = {
    "users":       "SELECT * FROM users WHERE id IN (SELECT user_id FROM submissions WHERE id IN ({ids})"
                   " UNION SELECT user_id FROM comments WHERE submission_id IN ({ids})) ORDER BY id",
    "forums":      "SELECT * FROM forums WHERE id IN (SELECT forum_id FROM submissions WHERE id IN ({ids})) ORDER BY id",
    "submissions": "SELECT * FROM submissions WHERE id IN ({ids}) ORDER BY id",
    "comments":    "SELECT * FROM comments WHERE submission_id IN ({ids}) ORDER BY id",
}


def copy_out(container: str, query: str) -> bytes:
    # explicit NULL marker: keeps NULL distinct from '' through CSV tools that drop quoting
    sql = f"COPY ({query}) TO STDOUT WITH (FORMAT csv, HEADER, NULL '\\N')"
    r = subprocess.run(["docker", "exec", container, "psql", "-U", "postmill", "-d", "postmill",
                        "-v", "ON_ERROR_STOP=1", "-c", sql], capture_output=True, check=True)
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the MemeEconomy environment patch")
    ap.add_argument("--split", choices=("paper98", "extension", "all"), default="paper98")
    ap.add_argument("--out", default=str(ROOT / "env_patch"))
    ap.add_argument("--container", default="reddit_benchmark")
    args = ap.parse_args()

    instances = json.loads(CONFIG_PATH.read_text())["instances"]
    tasks = json.loads(TASKS_PATH.read_text())["injected_tasks"]
    selected = {k: tasks[k] for k, v in instances.items()
                if (args.split == "all" or v.get("split") == args.split) and k in tasks}
    missing = [k for k, v in instances.items()
               if (args.split == "all" or v.get("split") == args.split) and k not in tasks]
    if missing:
        raise SystemExit(f"not injected: {missing}")
    ids = ",".join(str(int(t["submission_id"])) for t in selected.values())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "instances": len(selected), "files": {}}
    for table, query in TABLE_QUERIES.items():
        data = copy_out(args.container, query.format(ids=ids))
        path = out / f"{table}.csv.gz"
        with gzip.GzipFile(path, "wb", mtime=0) as f:   # mtime=0: byte-stable output
            f.write(data)
        count = int(subprocess.run(
            ["docker", "exec", args.container, "psql", "-U", "postmill", "-d", "postmill", "-tAc",
             f"SELECT COUNT(*) FROM ({query.format(ids=ids)}) q"],
            capture_output=True, text=True, check=True).stdout.strip())
        manifest["files"][path.name] = {"rows": count,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"  {table:12s} {count:6d} rows  → {path.name}")

    # tasks.json keeps only what the runner needs (submission id + URLs), not the config copy
    slim = {k: {kk: t[kk] for kk in ("submission_id", "url", "wiki_url") if kk in t}
            for k, t in selected.items()}
    (out / "tasks.json").write_text(json.dumps({"injected_tasks": slim}, indent=2))
    manifest["files"]["tasks.json"] = {"rows": len(slim),
                                       "sha256": hashlib.sha256((out / "tasks.json").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"exported {len(selected)} instances → {out}")


if __name__ == "__main__":
    main()
