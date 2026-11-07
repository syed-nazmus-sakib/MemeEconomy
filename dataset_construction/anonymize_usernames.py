#!/usr/bin/env python3
"""
anonymize_usernames.py
======================
Release step (provenance record). Replaces every real Reddit username in a staged
copy of the dataset with a stable pseudonym of the form user_XXXXXXXXXXX:

  * dis_mega.json   comment / post authors (`author`, `username`); u/<name> mentions and
                    @<name> mentions of dataset commenters in free text (public figures'
                    handles are kept); Reddit links to specific posts, comments, users or
                    share links become "[reddit link removed]" (markdown links keep their text);
                    `author_flair` is dropped
  * env_patch/      the matching Postmill accounts and the u/<name> mentions in comment
                    bodies. Accounts that also exist in the stock WebArena database
                    (created before the MemeEconomy injection) are not renamed — their
                    comments move to new pseudonymous accounts instead.

Pseudonyms are an HMAC of the username, so they are consistent across instances and
between the dataset and the environment, but cannot be reversed by hashing a list of
known usernames. The key is private and not distributed.

Comment text, scores, thread ids and timestamps are unchanged.

Usage:
    MEMEECONOMY_ANON_KEY=... python dataset_construction/anonymize_usernames.py \
        --dataset <staged>/memes/paper98 --patch <staged>/env_patch
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import string
from pathlib import Path

ANON_RE    = re.compile(r"^user_[A-Za-z0-9]{8,}$")
KEEP       = {"MemeTrackerBot", "_deleted_", "deleted", "[deleted]", "AutoModerator", "Anon"}
MENTION_RE = re.compile(r"(?<![\w/])(/?u/)([A-Za-z0-9_-]{3,20})")
AT_RE      = re.compile(r"(?<![\w@./])@([A-Za-z0-9_-]{3,25})")
LINK_RE    = re.compile(r"https?://(?:[a-z]+\.)?(?:reddit\.com|redd\.it)/[^\s)\]\"\\]+")
SPECIFIC   = re.compile(r"redd\.it/|/comments?/|/s/|/user/|/u/")
MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\(((?:https?://(?:[a-z]+\.)?(?:reddit\.com|redd\.it))?/[^)\s]*)\)")
INJECTION_START = "2026-01-01"   # accounts created earlier belong to the stock WebArena dump
NEW_ACCOUNT_IDS = 1_000_000      # above every user id in the stock WebArena dump (max 671,780)
ALPHABET   = string.digits + string.ascii_letters


def db_username(name: str) -> str:
    """The username inject_instances.py stores for a dataset author."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))[:25] or "Anon"


def _strip_link(m: re.Match) -> str:
    """Replace a link to a specific Reddit page; sentence punctuation after it stays."""
    url = m.group(0)
    trail = re.search(r"[.,;:!?]+$", url)
    core, tail = (url[:trail.start()], trail.group(0)) if trail else (url, "")
    return ("[reddit link removed]" if SPECIFIC.search(core) else core) + tail


class Pseudonymizer:
    def __init__(self, key: bytes, known: set[str]):
        self.key = key
        self.known = {db_username(n) for n in known}   # usernames of dataset commenters

    def needs(self, name: str) -> bool:
        return bool(name) and name not in KEEP and db_username(name) not in KEEP \
            and not ANON_RE.match(db_username(name))

    def __call__(self, name: str) -> str:
        if not self.needs(name):
            return name
        n = int.from_bytes(hmac.new(self.key, db_username(name).encode(), hashlib.sha256).digest(), "big")
        out = ""
        for _ in range(11):
            n, r = divmod(n, len(ALPHABET))
            out += ALPHABET[r]
        return "user_" + out

    def text(self, s: str) -> str:
        # markdown links to a specific Reddit page keep their text and lose the link
        s = MD_LINK_RE.sub(lambda m: m.group(1) if SPECIFIC.search(m.group(2)) else m.group(0), s)
        s = LINK_RE.sub(_strip_link, s)
        s = MENTION_RE.sub(lambda m: m.group(1) + self(m.group(2)), s)
        return AT_RE.sub(lambda m: "@" + self(m.group(1)) if db_username(m.group(1)) in self.known
                         else m.group(0), s)


def collect_authors(node, out: set) -> set:
    if isinstance(node, list):
        for x in node:
            collect_authors(x, out)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k in ("author", "username") and isinstance(v, str):
                out.add(v)
            else:
                collect_authors(v, out)
    return out


def anonymize_json(node, p: Pseudonymizer):
    if isinstance(node, list):
        return [anonymize_json(x, p) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "author_flair":
            continue
        if k in ("author", "username") and isinstance(v, str):
            out[k] = p(v)
        elif isinstance(v, str):                      # any text field may hold mentions or links
            out[k] = p.text(v)
        else:
            out[k] = anonymize_json(v, p)
    return out


def read_csv_gz(path: Path) -> tuple[list[str], list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv_gz(path: Path, fields: list[str], rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(buf.getvalue().encode("utf-8"))


def anonymize_patch(patch: Path, p: Pseudonymizer) -> dict:
    ufields, users = read_csv_gz(patch / "users.csv.gz")
    cfields, comments = read_csv_gz(patch / "comments.csv.gz")
    template = next(u for u in users if u["username"] == "MemeTrackerBot")
    next_id = max(NEW_ACCOUNT_IDS, max(int(u["id"]) for u in users) + 1)
    out_users, remap, renamed, moved = [], {}, 0, 0
    by_name: dict[str, dict] = {}
    for u in users:
        if not p.needs(u["username"]):
            out_users.append(u)
            continue
        alias = p(u["username"])
        if u["created"] < INJECTION_START:          # stock WebArena account: leave it, re-home comments
            if alias not in by_name:
                row = dict(template, id=str(next_id), username=alias,
                           normalized_username=alias.lower(), email=f"{alias.lower()}@example.com")
                by_name[alias] = row
                out_users.append(row)
                next_id += 1
            remap[u["id"]] = by_name[alias]["id"]
            moved += 1
        else:
            u.update(username=alias, normalized_username=alias.lower(), email=f"{alias.lower()}@example.com")
            out_users.append(u)
            renamed += 1
    names = [u["username"] for u in out_users]
    assert len(names) == len(set(names)), "pseudonym collision"
    for c in comments:
        c["user_id"] = remap.get(c["user_id"], c["user_id"])
        c["body"] = p.text(c["body"])
        c["search_doc"] = "\\N"                      # NULL; rebuilt by the comments_search_update trigger
    write_csv_gz(patch / "users.csv.gz", ufields, out_users)
    write_csv_gz(patch / "comments.csv.gz", cfields, comments)

    manifest = json.loads((patch / "manifest.json").read_text())
    manifest["files"]["users.csv.gz"]["rows"] = len(out_users)
    for name in manifest["files"]:
        manifest["files"][name]["sha256"] = hashlib.sha256((patch / name).read_bytes()).hexdigest()
    manifest["usernames"] = "pseudonymised for release (dataset_construction/anonymize_usernames.py)"
    (patch / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"renamed": renamed, "stock_accounts_rehomed": moved, "new_accounts": len(by_name)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pseudonymise Reddit usernames in a staged release")
    ap.add_argument("--dataset", required=True, help="staged instance_and_meme_dataset/")
    ap.add_argument("--patch", help="staged env_patch/")
    args = ap.parse_args()
    key = os.environ.get("MEMEECONOMY_ANON_KEY", "").encode()
    if len(key) < 16:
        raise SystemExit("set MEMEECONOMY_ANON_KEY (private, >= 16 chars)")
    files = sorted(Path(args.dataset).glob("*/dis_mega.json"))
    known: set[str] = set()
    for f in files:
        collect_authors(json.loads(f.read_text()), known)
    if args.patch:
        known |= {u["username"] for u in read_csv_gz(Path(args.patch) / "users.csv.gz")[1]}
    p = Pseudonymizer(key, {n for n in known if not ANON_RE.match(db_username(n)) and n not in KEEP})

    for f in files:
        f.write_text(json.dumps(anonymize_json(json.loads(f.read_text()), p), indent=2,
                                ensure_ascii=False), encoding="utf-8")
    print(f"dis_mega.json files anonymised: {len(files)}")
    if args.patch:
        print("env_patch:", anonymize_patch(Path(args.patch), p))


if __name__ == "__main__":
    main()
