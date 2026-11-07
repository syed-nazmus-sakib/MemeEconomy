#!/usr/bin/env python3
"""
inject_instances.py
===================
Injects MemeEconomy instances into the Postmill (Reddit WebArena) database running
in the `reddit_benchmark` Docker container: one MEGATHREAD submission per instance,
plus up to 25 comments taken from the instance's dis_mega.json.

Reads configs/instance_config.json and memes/ from the repo root.
Skips instances already listed in memeeconomy_tasks.json, so re-running is safe.
Writes the submission URLs to memeeconomy_tasks.json (injected_tasks).

Usually called by environment/setup_environment.py rather than directly.

Usage:
    python environment/inject_instances.py               # all splits
    python environment/inject_instances.py --split paper98
    python environment/inject_instances.py --dry-run
"""

from __future__ import annotations
import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "instance_config.json"
TASKS_PATH  = BASE_DIR / "memeeconomy_tasks.json"
DATASET_DIR = BASE_DIR / "memes"
CONTAINER   = "reddit_benchmark"

MAX_COMMENTS = 25   # maximum comments to inject per instance
random.seed(0)      # reproducible timestamps


# ── Docker SQL helpers ────────────────────────────────────────────────────────

def run_sql(sql: str, dry_run: bool = False) -> str | None:
    if dry_run:
        return "DRY_RUN_ID"
    cmd = [
        "docker", "exec", CONTAINER,
        "psql", "-U", "postmill", "-d", "postmill",
        "-t", "-A", "-q", "-c", sql,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [SQL ERR] {r.stderr.strip()[:200]}")
        return None
    return r.stdout.strip()


def escape(text) -> str:
    if text is None:
        return "NULL"
    return "'" + str(text).replace("'", "''") + "'"


def parse_id(output: str) -> str | None:
    if not output or output == "DRY_RUN_ID":
        return output
    for line in reversed(output.strip().splitlines()):
        if re.match(r"^\d+$", line.strip()):
            return line.strip()
    return None


def get_or_create_user(username: str, dry_run: bool) -> str | None:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", username)[:25] or "Anon"
    res = parse_id(run_sql(f"SELECT id FROM users WHERE username={escape(safe)}", dry_run))
    if res and res != "DRY_RUN_ID":
        return res
    email = f"{safe.lower()}@example.com"
    pw = "$2y$13$Vi.y.ne8h.X9/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g/g"
    sql = f"""INSERT INTO users (
        id, username, normalized_username, email, password, created,
        admin, locale, front_page, front_page_sort_mode, timezone,
        submission_link_destination, allow_private_messages
    ) VALUES (
        nextval('users_id_seq'), {escape(safe)}, {escape(safe.lower())}, {escape(email)},
        '{pw}', NOW(), FALSE, 'en', 'default', 'hot', 'UTC', 'url', TRUE
    ) RETURNING id"""
    return parse_id(run_sql(sql, dry_run))


def get_or_create_forum(forum_name: str, dry_run: bool) -> str | None:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", forum_name.removeprefix("r/"))
    res = parse_id(run_sql(f"SELECT id FROM forums WHERE name={escape(safe)}", dry_run))
    if res and res != "DRY_RUN_ID":
        return res
    sql = f"""INSERT INTO forums (
        id, name, normalized_name, title, description, sidebar, created, featured
    ) VALUES (
        nextval('forums_id_seq'), {escape(safe)}, {escape(safe.lower())}, {escape(safe)},
        {escape(f'The {safe} community.')}, '', NOW(), FALSE
    ) RETURNING id"""
    created = parse_id(run_sql(sql, dry_run))
    print(f"      + forum created: {safe} (id={created})")
    return created


# ── Comment helpers ───────────────────────────────────────────────────────────

def flatten_comments(items: list, depth: int = 0) -> list:
    """Flatten nested comments/replies into a single list (depth-first)."""
    flat = []
    for item in items:
        flat.append(item)
        replies = item.get("replies", item.get("children", []))
        if replies and depth < 3:
            flat.extend(flatten_comments(replies, depth + 1))
    return flat


def load_comments(folder: Path) -> list:
    """
    Load and flatten comments from dis_mega.json.
    Handles the three layouts in the dataset: {"post":..,"comments":[..]},
    {"megathread":{"posts":[..]}} and a bare list of comments.
    """
    mega = folder / "dis_mega.json"
    if not mega.exists():
        return []
    try:
        data = json.loads(mega.read_text())
    except Exception:
        return []

    if isinstance(data, list):                          # bare list
        return flatten_comments(data)
    if "comments" in data:                              # new format
        return flatten_comments(data["comments"])
    if "megathread" in data:                            # old format
        posts = data["megathread"].get("posts",
                data["megathread"].get("comments", []))
        return flatten_comments(posts)
    return []


def comment_fields(c: dict) -> tuple[str, str, int]:
    author = c.get("author", c.get("username", "Anon"))
    body   = c.get("body",   c.get("content", c.get("text", "...")))
    score_raw = c.get("score", c.get("upvotes", 1))
    if isinstance(score_raw, str):
        score_raw = score_raw.strip()
        if score_raw.upper().endswith("K"):
            try:
                score_raw = int(float(score_raw[:-1]) * 1000)
            except ValueError:
                score_raw = 1
        else:
            try:
                score_raw = int(score_raw)
            except ValueError:
                score_raw = 1
    return author, str(body), max(0, int(score_raw))


# ── Instance injection ────────────────────────────────────────────────────────

def inject_instance(
    instance_name: str,
    instance_data: dict,
    bot_user_id: str,
    dry_run: bool,
) -> dict | None:
    folder_name = instance_data.get("folder", instance_name)
    folder_path = DATASET_DIR / instance_data["split"] / folder_name

    # Determine forum from neu meme's subreddit
    memes = instance_data.get("memes", {})
    neu_sub = "OutOfTheLoop"
    for meme_id, meme in memes.items():
        if meme.get("category") == "neu":
            sub = meme.get("gt_subreddit", "OutOfTheLoop").removeprefix("r/")
            if sub:
                neu_sub = sub
            break

    forum_id = get_or_create_forum(neu_sub, dry_run)
    if not forum_id:
        print(f"    ✗ Could not create forum for {instance_name}")
        return None

    safe_forum = re.sub(r"[^a-zA-Z0-9_-]", "_", neu_sub)

    # Wiki page URL (served as static HTML from nginx)
    wiki_url = f"http://localhost:9999/news/{re.sub(r'[^a-zA-Z0-9_-]', '_', instance_name)}_news.html"

    ts_base = datetime.now() - timedelta(hours=random.randint(1, 72))
    ts_str  = ts_base.strftime("%Y-%m-%d %H:%M:%S")

    sub_sql = f"""INSERT INTO submissions (
        id, title, body, url, user_id, forum_id,
        timestamp, ip, sticky, ranking, moderated, user_flag, locked,
        last_active, comment_count, net_score, visibility, media_type
    ) VALUES (
        nextval('submissions_id_seq'),
        {escape(f'MEGATHREAD: {instance_name.replace("_"," ").title()}')},
        {escape(f'Discuss the recent events regarding {instance_name.replace("_"," ").title()} here.')},
        {escape(wiki_url)}, {bot_user_id}, {forum_id},
        '{ts_str}', '127.0.0.1', TRUE, 0, FALSE, 'none', FALSE,
        '{ts_str}', 0, 100, 'visible', 'url'
    ) RETURNING id"""
    sub_id = parse_id(run_sql(sub_sql, dry_run))
    if not sub_id:
        print(f"    ✗ Submission creation failed for {instance_name}")
        return None
    print(f"    ✓ submission #{sub_id} → {safe_forum} | wiki: {wiki_url}")

    # Inject comments
    comments = load_comments(folder_path)[:MAX_COMMENTS]
    count = 0
    cur_time = ts_base + timedelta(minutes=5)
    for c in comments:
        author, body, score = comment_fields(c)
        if not body.strip() or body == "...":
            continue
        author_id = get_or_create_user(author, dry_run)
        if not author_id:
            continue
        c_ts = cur_time.strftime("%Y-%m-%d %H:%M:%S")
        c_sql = f"""INSERT INTO comments (
            id, submission_id, user_id, body, timestamp,
            ip, moderated, visibility, user_flag, net_score
        ) VALUES (
            nextval('comments_id_seq'), {sub_id}, {author_id},
            {escape(body)}, '{c_ts}',
            '127.0.0.1', FALSE, 'visible', 'none', {score}
        )"""
        run_sql(c_sql, dry_run)
        count += 1
        cur_time += timedelta(minutes=random.randint(1, 20))

    print(f"    ✓ {count} comments injected")

    return {
        "submission_id": str(sub_id),
        "url": f"http://localhost:9999/f/{safe_forum}/{sub_id}",
        "wiki_url": wiki_url,
        "data": instance_data,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without touching the DB")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="instance config to read (default: configs/instance_config.json)")
    parser.add_argument("--split", choices=("paper98", "extension", "all"), default="all",
                        help="which instances to inject (default: all)")
    parser.add_argument("--container", default=CONTAINER, help=argparse.SUPPRESS)   # for testing
    args = parser.parse_args()
    globals()["CONTAINER"] = args.container
    dry_run = args.dry_run
    config_path = Path(args.config)

    if dry_run:
        print("[DRY-RUN MODE — no database writes]\n")

    # Load config
    config    = json.loads(config_path.read_text())
    print(f"Config: {config_path.name}")
    instances = config.get("instances", {})

    # Load existing tasks (to skip already-injected)
    existing_tasks: dict = {}
    if TASKS_PATH.exists():
        existing_tasks = json.loads(TASKS_PATH.read_text()).get("injected_tasks", {})
    already_done = {k for k, v in existing_tasks.items() if v.get("url", "#") != "#"}
    print(f"Existing injected tasks : {len(already_done)}")

    def folder_num(name: str) -> int:
        folder = instances[name].get("folder", "")
        m = re.match(r"^(\d+)_", folder)
        return int(m.group(1)) if m else 0

    new_instances = sorted(
        [n for n in instances
         if n not in already_done
         and (args.split == "all" or instances[n].get("split") == args.split)],
        key=folder_num,
    )
    print(f"New instances to inject : {len(new_instances)}")
    if not new_instances:
        print("Nothing to do.")
        return

    # The WebArena dump leaves the id sequences below max(id); without this,
    # nextval() collides with existing rows and inserts fail with duplicate keys.
    for table in ("users", "forums", "submissions", "comments"):
        run_sql(f"SELECT setval('{table}_id_seq', (SELECT MAX(id) FROM {table}))", dry_run)

    # Ensure MemeTrackerBot exists
    print("\nEnsuring MemeTrackerBot user...")
    bot_id = get_or_create_user("MemeTrackerBot", dry_run)
    print(f"  bot_id = {bot_id}")

    # Inject each new instance
    print("\nInjecting instances:\n" + "=" * 60)
    new_tasks = {}
    for name in new_instances:
        print(f"\n  [{folder_num(name)}] {name}")
        result = inject_instance(name, instances[name], bot_id, dry_run)
        if result:
            new_tasks[name] = result

    # Save updated tasks file
    if not dry_run:
        merged = {**existing_tasks, **new_tasks}
        out = {
            "generated_at": datetime.now().isoformat(),
            "injected_tasks": merged,
        }
        TASKS_PATH.write_text(json.dumps(out, indent=2))
        print(f"\n✓ memeeconomy_tasks.json updated: {len(merged)} total tasks")
    else:
        print(f"\n[DRY-RUN] Would write {len(new_tasks)} new tasks to memeeconomy_tasks.json")

    print(f"Done. Injected {len(new_tasks)} new instances.")


if __name__ == "__main__":
    main()
