#!/usr/bin/env python3
"""
setup_environment.py
====================
Turns a stock Reddit WebArena (Postmill) container into the MemeEconomy environment.

Prerequisites
  * the dataset is downloaded (python scripts/download_data.py)
  * the Reddit WebArena container is running as `reddit_benchmark` on port 9999, e.g.
        docker run --name reddit_benchmark -p 9999:80 -d postmill-populated-exposed-withimg

Steps
  1. add one MEGATHREAD (+ comments) per instance, either
       --exact : restore the paper's rows (original ids, usernames, timestamps) from
                 env_patch/, downloaded with the dataset             (recommended)
       default : inject fresh rows from the dataset files    environment/inject_instances.py
  2. build the Wikipedia-style news pages                   environment/generate_news_pages.py
  3. copy news pages and meme images into the container's web root
  4. check that a sample of pages, threads and images is served

Re-running is safe: existing rows and already-injected instances are skipped.

Usage:
    python environment/setup_environment.py --exact       # the paper's environment
    python environment/setup_environment.py --split all   # fresh injection, all instances
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
HERE        = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "instance_config.json"
TASKS_PATH  = ROOT / "memeeconomy_tasks.json"
DATASET_DIR = ROOT / "memes"
PAGES_DIR   = ROOT / "generated_pages"
PATCH_DIR   = ROOT / "env_patch"
CONTAINER   = "reddit_benchmark"
WEB_ROOT    = "/var/www/html/public"
BASE_URL    = "http://localhost:9999"
PATCH_TABLES = ("users", "forums", "submissions", "comments")   # FK order


def safe_name(instance_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", instance_name)


def docker(*args: str) -> None:
    subprocess.run(["docker", *args], check=True)


def check_prerequisites() -> None:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                       capture_output=True, text=True)
    if r.stdout.strip() != "true":
        sys.exit(f"[setup] container '{CONTAINER}' is not running — see the README (Environment).")
    if not DATASET_DIR.is_dir():
        sys.exit(f"[setup] {DATASET_DIR.name}/ not found — run: python scripts/download_data.py")


def psql(sql: str) -> str:
    r = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postmill", "-d", "postmill",
                        "-v", "ON_ERROR_STOP=1", "-tAc", sql], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def restore_patch(patch_dir: Path) -> None:
    """Insert the exported rows with their original ids; rows already present are kept."""
    manifest = json.loads((patch_dir / "manifest.json").read_text())
    for name, meta in manifest["files"].items():
        if hashlib.sha256((patch_dir / name).read_bytes()).hexdigest() != meta["sha256"]:
            sys.exit(f"[setup] checksum mismatch: {patch_dir / name}")

    tmp = "/tmp/memeeconomy_env_patch"
    docker("exec", CONTAINER, "mkdir", "-p", tmp)
    script = ["\\set ON_ERROR_STOP 1", "BEGIN;"]
    with tempfile.TemporaryDirectory() as local:
        for table in PATCH_TABLES:
            csv = Path(local) / f"{table}.csv"
            csv.write_bytes(gzip.decompress((patch_dir / f"{table}.csv.gz").read_bytes()))
            script += [f"CREATE TEMP TABLE patch_{table} (LIKE {table});",
                       f"\\copy patch_{table} FROM '{tmp}/{table}.csv' WITH (FORMAT csv, HEADER, NULL '\\N')",
                       f"INSERT INTO {table} SELECT * FROM patch_{table} ON CONFLICT DO NOTHING;"]
        # a pre-existing row with the same id must be the same entity (same name), or
        # threads would show the wrong author / forum
        script += ["SELECT 'conflict users ' || COUNT(*) FROM patch_users p JOIN users u USING (id)"
                   " WHERE u.username <> p.username;",
                   "SELECT 'conflict forums ' || COUNT(*) FROM patch_forums p JOIN forums f USING (id)"
                   " WHERE f.name <> p.name;"]
        script += [f"SELECT setval('{t}_id_seq', (SELECT MAX(id) FROM {t}));" for t in PATCH_TABLES]
        script.append("COMMIT;")
        (Path(local) / "restore.sql").write_text("\n".join(script) + "\n")
        docker("cp", f"{local}/.", f"{CONTAINER}:{tmp}/")
    r = subprocess.run(["docker", "exec", CONTAINER, "psql", "-U", "postmill", "-d", "postmill",
                        "-tA", "-f", f"{tmp}/restore.sql"], capture_output=True, text=True)
    docker("exec", CONTAINER, "rm", "-rf", tmp)
    if r.returncode != 0:
        sys.exit(f"[setup] restore failed:\n{r.stderr.strip()}")
    conflicts = [l for l in r.stdout.splitlines() if l.startswith("conflict") and not l.endswith(" 0")]
    if conflicts:
        sys.exit(f"[setup] base container has different rows under the paper's ids: {conflicts}")

    tasks =json.loads((patch_dir / "tasks.json").read_text())["injected_tasks"]
    sub_ids = ",".join(t["submission_id"] for t in tasks.values())
    found = int(psql(f"SELECT COUNT(*) FROM submissions WHERE id IN ({sub_ids})"))
    if found != len(tasks):
        sys.exit(f"[setup] only {found}/{len(tasks)} megathreads present after restore")

    existing = json.loads(TASKS_PATH.read_text()).get("injected_tasks", {}) if TASKS_PATH.exists() else {}
    TASKS_PATH.write_text(json.dumps({"injected_tasks": {**existing, **tasks}}, indent=2))
    print(f"[setup] restored {found} megathreads with their original ids")


def push_assets(instances: dict) -> None:
    """Copy news pages to test_pages/ and news/, meme images to newsfeed/<safe>/."""
    docker("exec", CONTAINER, "mkdir", "-p",
           f"{WEB_ROOT}/test_pages", f"{WEB_ROOT}/news", f"{WEB_ROOT}/newsfeed")
    with tempfile.TemporaryDirectory() as tmp:
        news, feed = Path(tmp) / "news", Path(tmp) / "newsfeed"
        news.mkdir()
        for name, data in instances.items():
            page = PAGES_DIR / f"{safe_name(name)}_news.html"
            if page.exists():
                shutil.copy2(page, news / page.name)
            dest = feed / safe_name(name)
            dest.mkdir(parents=True)
            for meme in data.get("memes", {}).values():
                src = DATASET_DIR / data["split"] / data["folder"] / meme["file"]
                if src.exists():
                    shutil.copy2(src, dest / meme["file"])
        docker("cp", f"{news}/.", f"{CONTAINER}:{WEB_ROOT}/test_pages/")
        docker("cp", f"{news}/.", f"{CONTAINER}:{WEB_ROOT}/news/")
        docker("cp", f"{feed}/.", f"{CONTAINER}:{WEB_ROOT}/newsfeed/")
    # docker cp keeps host permissions; macOS files arrive as 700 and nginx serves 403
    docker("exec", CONTAINER, "chmod", "-R", "755",
           f"{WEB_ROOT}/test_pages", f"{WEB_ROOT}/news", f"{WEB_ROOT}/newsfeed")


def verify(instances: dict, tasks: dict, sample: int = 5) -> bool:
    names = [n for n in instances if n in tasks][:sample]
    ok = True
    for name in names:
        data = instances[name]
        meme = next(iter(data["memes"].values()))["file"]
        for url in (f"{BASE_URL}/test_pages/{safe_name(name)}_news.html",
                    tasks[name]["url"].replace("http://localhost:9999", BASE_URL),
                    f"{BASE_URL}/newsfeed/{safe_name(name)}/{meme}"):
            try:
                status = urllib.request.urlopen(url, timeout=15).status
            except urllib.error.HTTPError as e:
                status = e.code
            except urllib.error.URLError as e:
                status = str(e.reason)
            if status != 200:
                ok = False
                print(f"  ✗ {status}  {url}")
    print(f"[setup] spot check of {len(names)} instances:", "OK" if ok else "FAILED")
    return ok


def main() -> None:
    global CONTAINER, BASE_URL
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact", action="store_true",
                    help="restore the paper's megathreads from env_patch/ instead of injecting fresh ones")
    ap.add_argument("--split", choices=("paper98", "extension", "all"), default="all",
                    help="instances to set up (default: all; the runner filters by split anyway)")
    ap.add_argument("--container", default=CONTAINER, help=argparse.SUPPRESS)   # for testing
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)    # for testing
    args = ap.parse_args()
    CONTAINER, BASE_URL = args.container, args.base_url

    check_prerequisites()
    py = sys.executable
    if args.exact:
        print("[setup] 1/4 restoring the paper's megathreads")
        restore_patch(PATCH_DIR)
        args.split = json.loads((PATCH_DIR / "manifest.json").read_text())["split"]
    else:
        print("[setup] 1/4 injecting megathreads")
        subprocess.run([py, str(HERE / "inject_instances.py"), "--split", args.split,
                        "--container", CONTAINER], check=True)
    print("[setup] 2/4 building news pages")
    subprocess.run([py, str(HERE / "generate_news_pages.py")], check=True)

    config = json.loads(CONFIG_PATH.read_text())["instances"]
    instances = {k: v for k, v in config.items()
                 if args.split == "all" or v.get("split") == args.split}
    print(f"[setup] 3/4 copying pages and images for {len(instances)} instances")
    push_assets(instances)

    print("[setup] 4/4 verifying")
    tasks = json.loads(TASKS_PATH.read_text()).get("injected_tasks", {})
    if not verify(instances, tasks):
        sys.exit(1)
    print("[setup] done — run: python -m memeeconomy.run_single_agent --models qwen8b --rounds 3")


if __name__ == "__main__":
    main()
