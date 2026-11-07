#!/usr/bin/env python3
"""
generate_karma_counts_v2.py
===========================
Generates karma_count.txt for instances 47–98.

Method
------
1. Read post.score from each instance's dis_mega.json.
2. Apply log-normalization across ALL new instances to get a 0–1 virality factor.
   virality_i = log(score_i + 1) / log(max_score + 1)
3. Map virality to per-category karma ranges:
     edgy1  : 3800–5300, with 7000 for top ~6% virality (≥ 0.92)
     edgy2  : 3500–4800
     edgy3  : 3200–4400
     funny1 : 3600–5000
     funny2 : 3200–4500
     funny3 : 3000–4100
     funny4 : 2800–3700
     funny5 : 2600–3400
     funny6 : 2500–3200
     funny7 : 2400–3000
     funny8 : 2200–2800
     neu1   : 1800–2500
     neu2   : 1500–2200
     poor1  : 200–1800  (nonlinear: virality^0.65 — very low for low-virality)
     poor2  : 100–1300  (nonlinear)
     poor3  : 100– 900  (nonlinear)
4. Round to nearest 100; format as "X.XK" / "XK" / plain number.
5. Assign subreddit by instance context (political/world/general) × category.
   For neu files: use the real subreddit from dis_mega.json post.subreddit.

No random values are used. Output is fully deterministic.
"""

import hashlib
import json
import math
import re
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "instance_and_meme_dataset"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}

# Category prefixes to recognise, sorted longest-first so "funny" beats "fun"
CAT_PREFIXES = ["edgy", "funny", "funy", "neu", "poor", "irr", "noise"]  # "funy" = typo of "funny"
TYPO_FIX = {"funy": "funny"}  # normalise before lookup

# ── Curated subreddit pools per category ────────────────────────────────────
SUBREDDIT_POOLS = {
    "funny": [
        "memes", "dankmemes", "sipstea", "madlads", "comedyheaven",
        "shitposting", "bossfights", "funnymemes", "holup", "PoliticalHumor",
        "politcialmeme", "greatbritishmemes", "oddlyspecific", "me_irl", "Fauxmoi",
    ],
    "poor": [
        "memes", "dankmemes", "sipstea", "madlads", "comedyheaven",
        "shitposting", "bossfights", "funnymemes", "holup", "PoliticalHumor",
        "politcialmeme", "greatbritishmemes", "oddlyspecific", "me_irl", "Fauxmoi",
        "PeterExplainsTheJoke", "explaintheJoke", "funny",
    ],
    "neu": [
        "usnews", "politics", "news", "worldnews", "todayilearned",
        "technology", "entertainment", "science", "television", "sports",
        "pics", "europe", "movies",
    ],
    "edgy": [
        "leopardsatemyface", "PublicFreakout", "theleftcantmeme", "therightcantmeme",
        "MurderedByWords", "facepalm", "memesopdidnotlike", "iamatotalpieceofshit",
        "WTF", "conservativememes", "antiwoke", "jokeofwoke", "persecutionfetish",
        "ShitLiberalsSay", "toiletpaperusa", "PoliticsPeopleTwitter",
        "fuckthealtright", "religiousfruitcake", "notliketheothergirls", "entitledbitch",
    ],
}
# irr/noise fall back to poor pool
SUBREDDIT_POOLS["irr"]   = SUBREDDIT_POOLS["poor"]
SUBREDDIT_POOLS["noise"] = SUBREDDIT_POOLS["poor"]


def pick_subreddit(instance_name: str, cat: str, cat_num: int) -> str:
    """Deterministic subreddit selection using MD5 hash of (instance, cat, num)."""
    pool = SUBREDDIT_POOLS.get(cat, SUBREDDIT_POOLS["poor"])
    key  = f"{instance_name}:{cat}:{cat_num}"
    idx  = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(pool)
    return f"r/{pool[idx]}"

# ── Karma ranges per (category, number) ─────────────────────────────────────
# (low, high, nonlinear)
# nonlinear=True → use virality^0.65 instead of virality (compresses low end)
KARMA_RANGES = {
    ("edgy",  1): (3800, 5300, False),
    ("edgy",  2): (3500, 4800, False),
    ("edgy",  3): (3200, 4400, False),
    ("funny", 1): (3600, 5000, False),
    ("funny", 2): (3200, 4500, False),
    ("funny", 3): (3000, 4100, False),
    ("funny", 4): (2800, 3700, False),
    ("funny", 5): (2600, 3400, False),
    ("funny", 6): (2500, 3200, False),
    ("funny", 7): (2400, 3000, False),
    ("funny", 8): (2200, 2800, False),
    ("neu",   1): (1800, 2500, False),
    ("neu",   2): (1500, 2200, False),
    ("poor",  1): ( 200, 1800, True),
    ("poor",  2): ( 100, 1300, True),
    ("poor",  3): ( 100,  900, True),
    ("irr",   1): ( 300,  700, False),
    ("irr",   2): ( 200,  600, False),
    ("noise", 1): ( 300,  700, False),
    ("noise", 2): ( 200,  600, False),
}
EDGY1_SPIKE_THRESHOLD = 0.92   # virality >= this → edgy1 gets 7000
EDGY1_SPIKE_VALUE     = 7000


# ─────────────────────────────────────────────────────────────────────────────

def get_meme_files(folder: Path) -> list[tuple[str, int, str]]:
    """Return sorted list of (normalised_cat, number, real_filename)."""
    memes = []
    for f in sorted(folder.iterdir()):
        real_name = f.name
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        # Strip macOS resource-fork prefixes and hidden files
        if real_name.startswith("._") or real_name.startswith("."):
            continue
        # Use stripped stem for category detection (handles " poor2.png" leading space)
        stem = f.stem.strip().lower()
        if any(stem.startswith(skip) for skip in ("webpage", "newsfeed", "screenshot")):
            continue
        for prefix in sorted(CAT_PREFIXES, key=len, reverse=True):
            if stem.startswith(prefix):
                rest = stem[len(prefix):]
                if rest == "":
                    number = 1
                elif rest.isdigit():
                    number = int(rest)
                else:
                    continue  # e.g. "edgysuper" — not a match
                norm_cat = TYPO_FIX.get(prefix, prefix)
                memes.append((norm_cat, number, real_name))
                break
    # Sort by (category, number) so output is deterministic
    return sorted(memes, key=lambda t: (t[0], t[1]))


def get_post_score(folder: Path) -> int:
    mega = folder / "dis_mega.json"
    if mega.exists():
        try:
            data = json.loads(mega.read_text())
            return int(data.get("post", {}).get("score", 0))
        except Exception:
            pass
    return 0


def compute_karma(cat: str, num: int, virality: float) -> int:
    # Edgy1 spike: very rare top instances get 7K
    if cat == "edgy" and num == 1 and virality >= EDGY1_SPIKE_THRESHOLD:
        return EDGY1_SPIKE_VALUE

    key = (cat, num)
    if key in KARMA_RANGES:
        low, high, nonlinear = KARMA_RANGES[key]
    else:
        # Fallback for any higher numbered categories
        low, high, nonlinear = 400, 1200, False

    v = (virality ** 0.65) if nonlinear else virality
    raw = low + v * (high - low)
    return max(low, round(raw / 100) * 100)


def format_upvotes(n: int) -> str:
    if n >= 1000:
        k = n / 1000
        return f"{int(k)}K" if k == int(k) else f"{k:.1f}K"
    return str(n)


def main():
    # ── Collect all new instances (47–98) with their post scores ────────────
    instances = []
    for d in sorted(
        DATASET_DIR.iterdir(),
        key=lambda p: int(re.match(r'^(\d+)', p.name).group(1))
                      if re.match(r'^(\d+)', p.name) else 9999,
    ):
        if not d.is_dir():
            continue
        m = re.match(r'^(\d+)_(.*)', d.name)
        if not m:
            continue
        num = int(m.group(1))
        if num < 47:
            continue
        name = m.group(2)
        score = get_post_score(d)
        instances.append((num, name, d, score))

    if not instances:
        print("No new instances (47+) found.")
        return

    # ── Log-normalise scores to get virality in [0, 1] ──────────────────────
    scores = [s for _, _, _, s in instances]
    max_score = max(scores)
    max_log = math.log(max_score + 1)
    print(f"Instances : {len(instances)}  (#{instances[0][0]}–#{instances[-1][0]})")
    print(f"Score range: {min(scores):,} – {max_score:,}")
    print(f"log-max   : {max_log:.4f}")
    print(f"7K spike threshold: virality >= {EDGY1_SPIKE_THRESHOLD}")
    print()

    total_lines = 0
    for num, name, folder, score in instances:
        virality = math.log(score + 1) / max_log

        memes = get_meme_files(folder)
        if not memes:
            print(f"  WARN  #{num} {name}: no meme images found — skipped")
            continue

        lines = []
        for cat, cat_num, real_filename in memes:
            karma = compute_karma(cat, cat_num, virality)
            sub   = pick_subreddit(name, cat, cat_num)
            lines.append(f"{real_filename} - {format_upvotes(karma)} upvotes subreddit {sub}")

        out = folder / "karma_count.txt"
        out.write_text("\n".join(lines) + "\n")
        spike = "  *** 7K SPIKE ***" if virality >= EDGY1_SPIKE_THRESHOLD else ""
        print(f"  #{num:3d}  {name[:50]:<50}  virality={virality:.3f}  → {len(lines)} entries{spike}")
        total_lines += len(lines)

    print(f"\nDone.  {total_lines} total karma entries written across {len(instances)} instances.")


if __name__ == "__main__":
    main()
