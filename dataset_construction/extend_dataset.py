#!/usr/bin/env python3
"""
extend_dataset.py
=================
Builds instances 99–300 from the raw material in MemeECON/ and meme_part3/ and
appends them to the benchmark dataset + config.

Steps
-----
1. Enumerate MemeECON/instances/{post 98, post 150, post 170, post 217}
   (first 46(partial) and post46 duplicate existing 1–98 and are ignored).
   A folder in MemeECON/Update/edgy fix/ with the same name replaces its raw
   folder (same memes + the missing edgy one).
2. Assign numbers: 99–170 come from the folder prefix; post 170 → 171–217 and
   post 217 → 218–230 in sorted order; 231–300 follow ORDER_231_300 (meme_part3/,
   MemeECON/Update/revive/, news from hello.txt, megathreads from MemeECON/Update/megathread/).
3. Copy memes into instance_and_meme_dataset/<N>_<slug>/ with normalised
   filenames (edgy.jpg → edgy1.jpg, neutral.webp → neu1.webp, "funnny" → funny,
   leading spaces stripped, "ttt" folder prefix removed).
4. Copy the matching megathread JSON as dis_mega.json.
5. For 99–170, write news_article.txt from 99_170.txt (title / dateline / body).
6. Generate karma_count.txt deterministically from megathread post.score
   (log-virality, same method as generate_karma_counts_v2.py) with
   taxonomy-consistent subreddits (same context rules as generate_karma_counts.py).
7. Write instance_config_300.json = original 98 entries verbatim (+ split tag) +
   202 new entries in the same schema (base_karma normalised so max per instance = 1000).
   Every instance carries split = paper98 | extension | pending; pending instances are
   incomplete (see their "issues") and must not be run.
8. Write MemeECON/extension_manifest.json documenting every decision.

The original instance_config.json and folders 1–98 are never modified.

Usage:
    python3 extend_dataset.py            # build
    python3 extend_dataset.py --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
RAW         = ROOT / "MemeECON"
DATASET_DIR = ROOT / "instance_and_meme_dataset"
CONFIG_IN   = ROOT / "instance_config.json"
CONFIG_OUT  = ROOT / "instance_config_300.json"
NEWS_TXT    = ROOT / "99_170.txt"
NEWS_TXT_2  = ROOT / "News_Summary_171-230.txt"      # "N. Title" + body, no dateline
NEWS_TXT_3  = ROOT / "hello.txt"                     # "N. Title (Date)" + body; "## N. Title" for revive
UPDATE      = RAW / "Update"
EDGY_FIX    = UPDATE / "edgy fix"
NEWS_FIXES  = UPDATE / "news_fixes.txt"              # replacements for hello.txt articles about the wrong event
PART3       = ROOT / "meme_part3"
MEGA_DIR_3  = UPDATE / "megathread"
# Hand-off folder for instances still being collected: 19_missing/<N>_<key>/ holds that instance's
# memes, megathread.json and news_article.txt and, when present, replaces every other source for it.
# WRONG_megathread_*.json files there are ignored.
MISSING_DIR = ROOT / "19_missing"

# 231–300, in hello.txt order (1–68, then revive "## 48"/"## 49").
# (key, source folder, megathread file | None, domain, context, word that must appear in the hello.txt title)
P3, NE, RV = PART3, PART3 / "newentry", UPDATE / "revive"
POP = ("pop_culture", "pop_culture")
ORDER_231_300 = [
    ("sam_mendes_1917",                       NE / "1917",                "1917.json",                *POP, "1917"),
    ("la_wildfires_2025",                     P3 / "la_fires",            "LA_fires.json",            "event", "general", "Wildfires"),
    ("spider_man_across_the_spider_verse",    NE / "acrossspiderverse",   "acrossspiderverse.json",   *POP, "Across the Spider-Verse"),
    ("avatar_the_way_of_water",               NE / "avatarwater",         "avatarwater.json",         *POP, "Avatar"),
    ("baby_reindeer",                         P3 / "babyraindeer",        "babyraindeer.json",        *POP, "Baby Reindeer"),
    ("the_banshees_of_inisherin",             NE / "bansheesofinsherin",  "bansheesofinsherin.json",  *POP, "Banshees"),
    ("barbenheimer",                          P3 / "barbenheimer",        "barbenheimer.json",        *POP, "Barbenheimer"),
    ("kanye_west_and_bianca_censori_outfits", P3 / "bianca_kanye",        "bianca_kanye.json",        "event", "general", "Bianca"),
    ("biden_drops_out_of_2024_race",          P3 / "biden drop",          "biden drop.json",          "event", "political", "Biden"),
    ("bing_ai_chatbot_rogue",                 P3 / "bing",                "bing.json",                "event", "general", "Bing"),
    ("black_panther",                         NE / "black_panther",       "black_panther.json",       *POP, "Black Panther"),
    ("bohemian_rhapsody",                     NE / "bohemianrapsody",     "bohemianrapsody.json",     *POP, "Bohemian"),
    ("charli_xcx_brat",                       P3 / "brat_xcx",            "brat_xcx.json",            *POP, "Brat"),
    ("hbo_chernobyl",                         NE / "chernobyl",           "chernobyl.json",           *POP, "Chernobyl"),
    ("chill_guy_meme",                        P3 / "chill_guy",           "chill_guy.json",           *POP, "Chill Guy"),
    ("beyonce_cowboy_carter",                 P3 / "cowboycarter",        "cowboycarter.json",        *POP, "Cowboy Carter"),
    ("deepseek_ai_launch",                    P3 / "deepseek",            "deepseek.json",            "event", "general", "DeepSeek"),
    ("diddy_arrest",                          P3 / "diddy_arrest",        "diddy_arrest.json",        "event", "general", "Diddy"),
    ("drive_ryan_gosling_literally_me",       NE / "drive_meme",          "drive_meme.json",          *POP, "Drive"),
    ("new_jersey_drone_sightings",            P3 / "droneLA",             "la drone.json",            "event", "general", "Drone"),
    ("dubai_floods_2024",                     P3 / "dubai_flood",         "dubai_flood.json",         "event", "world", "Dubai"),
    ("dune_2021",                             NE / "dune1",               "dune1.json",               *POP, "Dune"),
    ("everything_everywhere_all_at_once",     NE / "eeaao",               "eeaao.json",               *POP, "Everything Everywhere"),
    ("elon_musk_chainsaw",                    P3 / "elon_chainsaw",       "elon_chainsaw.json",       "event", "political", "Chainsaw"),
    ("sicario",                               NE / "endgame",             "endgame.json",             *POP, "Endgame"),
    ("killers_of_the_flower_moon",            NE / "flowermoon",          "flowermoon.json",          *POP, "Flower Moon"),
    ("george_santos_expelled",                P3 / "georgesantos",        "georgesantos.json",        "event", "political", "Santos"),
    ("grimace_shake_trend",                   P3 / "grimace_shake",       "grimace_shake.json",       *POP, "Grimace"),
    ("im_just_ken",                           P3 / "imken",               "imken.json",               *POP, "Ken"),
    ("avengers_infinity_war",                 NE / "infinity_war",        "infinity_war.json",        *POP, "Infinity War"),
    ("ja_morant_gun_suspension",              P3 / "jamorantgun",         "jamorantgun.json",         "event", "general", "Morant"),
    ("jd_vance_couch_rumor",                  P3 / "jdvancecouch",        "jdvancecouch.json",        "event", "political", "Vance"),
    ("joker_2019",                            NE / "joker",               "joker.json",               *POP, "Joker"),
    ("katy_perry_blue_origin_flight",         P3 / "katyspace",           "katyspace.json",           "event", "general", "Katy Perry"),
    ("kevin_mccarthy_ousted_as_speaker",      P3 / "kevin_speaker",       "kevin speaker.json",       "event", "political", "McCarthy"),
    ("king_charles_coronation",               P3 / "king_charles",        "king_charles.json",        "event", "world", "King Charles"),
    ("lauren_boebert_beetlejuice_ejection",   P3 / "lauren",              "lauren.json",              "event", "political", "Boebert"),
    ("luigi_mangione_arrest",                 P3 / "luigi",               None,                       "event", "general", "Luigi"),
    ("macron_brigitte_airplane_shove",        P3 / "macron_wife_slap",    "macron_wife_slap.json",    "event", "world", "Macron"),
    ("moo_deng",                              P3 / "moodeng",             "moodeng.json",             *POP, "Moo Deng"),
    ("jake_paul_vs_mike_tyson",               P3 / "mtvsjp",              "mtvsjp.json",              "event", "general", "Tyson"),
    ("musk_vs_zuckerberg_cage_match",         P3 / "muskvszucc",          "muskvszucc.json",          "event", "general", "Zuckerberg"),
    ("no_time_to_die",                        NE / "notimetodie",         "notimetodie.json",         *POP, "No Time to Die"),
    ("shohei_ohtani_interpreter_gambling",    P3 / "ohtani_gamble",       "ohtani_gamble.json",       "event", "general", "Ohtani"),
    ("once_upon_a_time_in_hollywood",         NE / "onceuponhollywood",   "onceuponhollywood.json",   *POP, "Once Upon"),
    ("palm_springs_film",                     NE / "palmspring",          "palmspring.json",          *POP, "Palm Springs"),
    ("parasite_oscars",                       NE / "parasite",            "parasite.json",            *POP, "Parasite"),
    ("prigozhin_plane_crash",                 P3 / "prigozhin",           "prigozhin.json",           "event", "world", "Prigozhin"),
    ("a_quiet_place",                         NE / "quietplace",          "quietplace.json",          *POP, "Quiet Place"),
    ("rednote_tiktok_refugees",               P3 / "rednote",             "rednote.json",             "event", "general", "Rednote"),
    ("pixar_soul",                            NE / "soul",                "soul.json",                *POP, "Soul"),
    ("spider_man_no_way_home",                NE / "spiderman_nowayhome", "spiderman_nowayhome.json", *POP, "No Way Home"),
    ("spider_man_into_the_spider_verse",      NE / "spiderverse1",        "spiderverse1.json",        *POP, "Into the Spider-Verse"),
    ("boeing_starliner_stranded_astronauts",  P3 / "starliner",           "starliner.json",           "event", "general", "Starliner"),
    ("succession_finale",                     P3 / "succession_finale",   "succession_finale.json",   *POP, "Succession"),
    ("super_mario_bros_movie",                NE / "supermario",          "supermario.json",          *POP, "Super Mario"),
    ("taylor_swift_engagement",               P3 / "taylorswift",         "taylorswift.json",         "event", "general", "Taylor Swift"),
    ("tenet",                                 NE / "tenet",               "tenet.json",               *POP, "Tenet"),
    ("the_batman",                            NE / "thebatman",           "thebatman.json",           *POP, "Batman"),
    ("threads_launch",                        P3 / "threads_launch",      "threads_launch.json",      "event", "general", "Threads"),
    ("top_gun_maverick",                      NE / "topgunm",             "topgunm.json",             *POP, "Top Gun"),
    ("trump_mamdani_oval_office_meeting",     P3 / "trump+mamdani",       "trump+mamdani.json",       "event", "political", "Mamdani"),
    ("trump_zelensky_oval_office_clash",      P3 / "trump_zelensky",      "trump_zelensky.json",      "event", "political", "Zelensky"),
    ("trump_hush_money_verdict",              P3 / "trumphushmny",        "trumphushmny.json",        "event", "political", "Hush Money"),
    ("twitter_rebrands_to_x",                 P3 / "twitter2x",           "twitter2x.json",           "event", "general", "Twitter"),
    ("wandavision",                           NE / "wandavision",         "wandavision.json",         *POP, "WandaVision"),
    ("wicked_2024",                           P3 / "wicked",              "wicked.json",              *POP, "Wicked"),
    ("yellowstone",                           NE / "yellowstone",         "yellowstone.json",         *POP, "Yellowstone"),
    ("judge_holden",                          RV / "judgeholden",         None,                       *POP, "Judge Holden"),
    ("scp_173",                               RV / "scp173",              None,                       *POP, "SCP-173"),
]
RAW_PREFIXES = {"scp173": "173scp"}                      # raw folder slug → filename prefix to drop
FILE_RELABEL = {("sam_mendes_1917", "edgy.jpg"): "funny",          # caption-only "soulmates don't exi-" joke, not Tier 5
                ("jake_paul_vs_mike_tyson", "neutral(1).jpeg"): "neu"}  # second neutral → neu2
IGNORE_FILES = {("trump_mamdani_oval_office_meeting", "sddefault.jpg"),  # PBS News video thumbnail, not a meme
                ("elon_musk_heil", "funny3(1).png"),                    # byte-identical copy of funny3.png
                ("monster_romance_book", "funny1 (1).jpg")}             # see WARNINGS

# Known problems that keep an instance out of the runnable set (split = pending) until fixed.
# Missing edgy / <4 memes / no megathread / no news are detected automatically.
# A plain message applies only while the instance is built from its original source; once it is
# built from 19_missing/ (where the wrong megathread is renamed WRONG_* and ignored), only the
# (message, still_applies(folder)) entries in BLOCKER_CHECKS are re-checked.
BLOCKER_CHECKS = {
    "sicario":                   ("old Avengers: Endgame files (duplicate of #179) still in the folder",
                                  lambda d: (d / "NEW_EVENT.txt").exists()
                                  or any((d / f).exists() for f in ("poor.avif", "funny.jpeg", "edgy.jpeg", "neutral.jpeg"))),
    "dune_2021":                 ("2 files are the same image",
                                  lambda d: (d / "k5kmcen3496b1.jpg").exists() and
                                  (d / "we-need-more-of-this-specific-brand-of-dune-meme-v0-k5kmcen3496b1.webp").exists()),
}
# Slots whose original event was replaced (key renamed), so the hello.txt article no longer applies.
REPLACED_EVENTS = {"sicario"}  # was avengers_endgame_premiere (duplicate of #179)
BLOCKERS = {
    "bing_ai_chatbot_rogue":            "megathread bing.json is a copy of the Biden-drops-out thread",
    "katy_perry_blue_origin_flight":    "megathread katyspace.json is about the Kate Spade handbag brand",
    "palm_springs_film":                "megathread palmspring.json is the Palm Springs city monthly thread, not the film",
    "super_mario_bros_movie":           "megathread supermario.json is about The Super Mario Galaxy Movie (2026), not the 2023 film",
    "wicked_2024":                      "megathread wicked.json (Nov 2025) is about the sequel Wicked: For Good",
    "a_quiet_place":                    "megathread quietplace.json is about A Quiet Place: Day One (2024), not the 2018 film",
    "sicario":                          "duplicate of #179 avengers_endgame (same event; poor.avif byte-identical to its poor1.jpg, funny.jpeg = its neu1.jpg)",
    "dune_2021":                        "memes unlabelled; 2 of the 5 files are the same image",
}
# Usable, but worth fixing (megathread is a related but different event, or very thin).
WARNINGS = {
    "deepseek_ai_launch":                   "megathread is a Sept 2026 r/DeepSeek complaint post, not the Jan 2025 launch",
    "diddy_arrest":                         "megathread is the May 2025 trial thread, not the Sept 2024 arrest",
    "drive_ryan_gosling_literally_me":      "megathread is a 2016 r/tipofmytongue post with 11 comments",
    "george_santos_expelled":               "megathread is about the May 2023 charges, not the Dec 2023 expulsion",
    "king_charles_coronation":              "megathread is the Sept 2022 accession thread (overlaps #58 king_charles_sausage_fingers), not the coronation",
    "boeing_starliner_stranded_astronauts": "megathread is a Sept 2026 follow-up, not the Sept 2024 return",
    "taylor_swift_engagement":              "megathread is a 2026 wedding discussion",
    "trump_mamdani_oval_office_meeting":    "megathread is a 2026 press-ban thread",
    "trump_zelensky_oval_office_clash":     "megathread is an Oct 2026 thread about diesel prices",
    "yellowstone":                          "news is the Jan 2022 season-4 finale; megathread is Dec 2024 series-finale predictions",
    "ja_morant_gun_suspension":             "megathread is a 2026 retrospective",
    "luigi_mangione_arrest":                "megathread is the Dec 2025 r/FreeLuigi court-hearing thread, not the Dec 2024 arrest",
    "super_mario_bros_movie":               "megathread is the Oct 2022 trailer thread, not the Apr 2023 release",
    "trump_hush_money_verdict":             "neu1 (Cheeto 'leaked photo') is a sexual joke; kept as neutral by decision 2026-11-05",
    "monster_romance_book":                 "raw 'funny1 (1).jpg' was never included (filename not parsed); the instance has no funny1",
}

# Raw folders excluded from the dataset entirely (no megathread AND no news article).
EXCLUDE_SLUGS = {"adam_rose_pb_j_bites_instagram_ad"}

# Canonical numbering for the unnumbered batches (post 170 / post 217), taken from
# News_Summary_171-230.txt. Slugs not listed here (currently only Adam Rose, which has
# neither a megathread nor an article) are appended after 230.
ORDER_171_230 = [
    "7x7_49_meme", "67_meme", "alien_glorp", "american_dad", "amhate", "amogus", "aura", "avatar",
    "avengers_endgame", "baldi", "berserk", "billy_and_mandy_memes", "breaking_bad", "chickenjocky",
    "courage", "danny_trejo", "dexter", "django_unchained", "dvd", "gekyume", "gravityfalls", "gunther",
    "hello_neighbour", "house_md", "humancentipede", "hunchback_of_notredame", "jpmorgan", "klaus_schwab",
    "koe_no_katachi", "lamar", "lavagirl_and_sharkboy", "lkmx", "malcolm", "monster_romance_book",
    "nicijo", "obamaprism", "pirates_of_the_carribean", "qu", "rango", "scp096", "shadowlatina", "sigma",
    "skinwalker", "stranger_thing_ending_conspiracy", "tarantino_feet", "tung_tung_sahur", "young_sheldon",
    "erling_haaland_s_majin_buu_era", "supreme_leader_kylian_mbappe", "erling_haaland_running",
    "sophie_cunningham_pointing", "kane_parsons_he_is_20_year_old", "will_smith_presenting",
    "essie_world_cup_curse", "reginald_skulinski_eating_disappointed_monster_house",
    "sabrina_carpenter_yodeling", "japan_is_turning_footsteps_into_electricity",
    "the_kate_middleton_photo_editing_scandal", "lindsay_clancy_trial",
    "robert_pattinson_somebody_get_these_beggars",
]
MANIFEST    = RAW / "extension_manifest.json"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".jfif"}
EXT_MAP    = {".jfif": ".jpg"}                         # .jfif is plain JPEG; browsers/MIME know .jpg

# batch folder → (numbering mode, domain)
BATCHES = [
    ("post 98",  "prefixed", "event"),
    ("post 150", "prefixed", "event"),
    ("post 170", "sequential", "pop_culture"),
    ("post 217", "sequential", "event"),
]
MEGA_DIRS = [RAW / "megathread" / "99-170", RAW / "megathread" / "171-230"]

# instance folder (raw name) → megathread file, where fuzzy matching is wrong/ambiguous
MEGA_OVERRIDES = {   # keyed by slug (raw names may carry accents / NFD normalisation)
    "donald_trump_trans_weightlifter_impression_ahegao_face": "Trump ahegao.json",
    "supreme_leader_kylian_mbappe": "dictator mbappe.json",
    "tung_tung_sahur": "tung tung sahur.json",
    "adam_rose_pb_j_bites_instagram_ad": None,   # no megathread exists
}
SLUG_OVERRIDES = {"67": "67_meme", "7x7_49": "7x7_49_meme", "ttt": "tung_tung_sahur"}

TIER = {
    "edgy":  (5, "Edgy/Offensive"),
    "funny": (1, "Funny/Good Humor"),
    "neu":   (3, "Neutral/Descriptive"),
    "poor":  (2, "Poor Execution"),
    "irr":   (4, "Irrelevant"),
    "noise": (4, "Irrelevant"),
}
CAT_ALIASES = {"neutral": "neu", "neutal": "neu", "nrutral": "neu", "funnny": "funny", "funy": "funny"}
CAT_PREFIXES = ["funnny", "neutral", "neutal", "nrutral", "funny", "edgy", "noise", "poor", "funy", "neu", "irr"]

# ── karma (identical ranges to generate_karma_counts_v2.py) ─────────────────
KARMA_RANGES = {
    ("edgy", 1): (3800, 5300, False), ("edgy", 2): (3500, 4800, False), ("edgy", 3): (3200, 4400, False),
    ("funny", 1): (3600, 5000, False), ("funny", 2): (3200, 4500, False), ("funny", 3): (3000, 4100, False),
    ("funny", 4): (2800, 3700, False), ("funny", 5): (2600, 3400, False), ("funny", 6): (2500, 3200, False),
    ("funny", 7): (2400, 3000, False), ("funny", 8): (2200, 2800, False),
    ("neu", 1): (1800, 2500, False), ("neu", 2): (1500, 2200, False),
    ("poor", 1): (200, 1800, True), ("poor", 2): (100, 1300, True), ("poor", 3): (100, 900, True),
    ("irr", 1): (300, 700, False), ("irr", 2): (200, 600, False),
    ("noise", 1): (300, 700, False), ("noise", 2): (200, 600, False),
}
EDGY1_SPIKE_THRESHOLD, EDGY1_SPIKE_VALUE = 0.92, 7000

# ── context classification (drives subreddit assignment) ────────────────────
POLITICAL_KW = [
    "trump", "biden", "obama", "hillary", "clinton", "bernie", "sanders", "romney", "bush",
    "kamala", "harris", "vance", "patel", "election", "parliament", "ice", "kirk", "giuliani",
    "rubio", "cruz", "desantis", "pelosi", "9_11", "somali", "daycare", "greenland", "kimmel",
    "covington", "rent_is_too_damn_high", "binders", "horses_and_bayonets", "delete_your_account",
    "four_seasons", "we_did_it_joe", "will_you_shut_up", "worst_trade_deal", "baby_balloon",
    "jpmorgan", "klaus_schwab", "charlie_kirk", "norwegian_tourist", "minneapolis",
]
WORLD_KW = [
    "french", "france", "uk_", "online_safety", "pope", "conclave", "norway", "japan", "world_cup",
    "mbappe", "mbapp", "haaland", "neymar", "kate_middleton", "essie", "bonnie_blue", "macron",
    "cracker_barrel", "jaguar", "tea_app", "cardinal", "united_airlines", "oceangate", "raygun",
]
CONTEXT_OVERRIDES = {   # slug → context
    "elon_musk_and_donald_trump_breakup": "political",
    "elon_musk_heil": "political",
    "george_bush_9_11": "political",
    "lindsay_clancy_trial": "general",
    "united_airlines_passenger_removal": "general",
    "jaguar_rebrand": "general",
    "cracker_barrel_logo_change": "general",
    "tea_app_dataleak": "general",
    "oceangate_submerine": "world",
}

# neutral/niche subs the karma taxonomy already knows (derived from instance_config.json at runtime)
NEU_TAXONOMY_SUBS: set[str] = set()


# ── helpers ──────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"\.\s*x$", "", name.strip())
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


def norm_match(s: str) -> str:
    s = s.lower()
    s = re.sub(r"^\d+_", "", s); s = re.sub(r"\.json$", "", s); s = re.sub(r"\.\s*x$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\b(megathread|the|of|a|an|and|in|at|his|her|its|image)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def list_dirs(p: Path):
    return sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("._"))


def list_images(folder: Path):
    return [f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            and not f.name.startswith(".") and not f.name.startswith("._")]


def parse_meme_name(stem: str, folder_slug: str) -> tuple[str, int | None] | None:
    """Return (category, number|None) from a raw filename stem, or None if not a meme."""
    s = stem.strip().lower()
    s = re.sub(r"[\s_\-]+", "", s)
    if folder_slug in RAW_PREFIXES and s.startswith(RAW_PREFIXES[folder_slug]):
        s = s[len(RAW_PREFIXES[folder_slug]):]
    if s.startswith(("webpage", "newsfeed", "screenshot")):
        return None
    # strip a folder-name prefix like "ttt" in "tttedgy"
    for pfx in sorted(CAT_PREFIXES, key=len, reverse=True):
        idx = s.find(pfx)
        if idx == 0 or (idx > 0 and folder_slug.startswith(s[:idx])):
            rest = s[idx + len(pfx):]
            cat = CAT_ALIASES.get(pfx, pfx)
            if rest == "":
                return cat, None
            if rest.isdigit():
                return cat, int(rest)
            return None
    return None


def normalise_memes(folder: Path, folder_slug: str, key: str | None = None):
    """Return list of (new_filename, category, number, src_path, old_filename)."""
    parsed = []
    for f in list_images(folder):
        if (key, f.name) in FILE_RELABEL:
            parsed.append((FILE_RELABEL[(key, f.name)], None, f))
            continue
        r = parse_meme_name(f.stem, folder_slug)
        if r is None:
            continue
        parsed.append((r[0], r[1], f))
    used: set[tuple[str, int]] = set()
    out = []
    # numbered first so unnumbered files take the next free slot
    for cat, num, f in sorted(parsed, key=lambda t: (t[0], t[1] is None, t[1] or 0)):
        n = num if num is not None else 1
        while (cat, n) in used:
            n += 1
        used.add((cat, n))
        ext = EXT_MAP.get(f.suffix.lower(), f.suffix.lower())
        out.append((f"{cat}{n}{ext}", cat, n, f, f.name))
    return sorted(out, key=lambda t: (t[1], t[2]))


def context_for(slug: str, domain: str) -> str:
    if slug in CONTEXT_OVERRIDES:
        return CONTEXT_OVERRIDES[slug]
    if domain == "pop_culture":
        return "pop_culture"
    if any(k in slug for k in POLITICAL_KW):
        return "political"
    if any(k in slug for k in WORLD_KW):
        return "world"
    return "general"


def subreddits_for(cat: str, ctx: str, neu_gt_sub: str | None):
    """Return (gt_subreddit, primary_subreddit, all_fit_subreddits) mirroring instance_config.json patterns."""
    if cat == "edgy":
        if ctx == "political":
            return "r/PoliticalHumor", "r/PoliticalHumor", ["r/PoliticalHumor", "r/dankmemes"]
        return "r/dankmemes", "r/dankmemes", ["r/dankmemes", "r/shitposting"]
    if cat == "funny":
        if ctx == "political":
            return "r/PoliticalHumor", "r/PoliticalHumor", ["r/PoliticalHumor", "r/memes"]
        return "r/memes", "r/memes", ["r/memes", "r/funny"]
    if cat == "neu":
        gt = f"r/{neu_gt_sub}" if neu_gt_sub else None
        if neu_gt_sub in NEU_TAXONOMY_SUBS:
            primary = gt
        else:
            primary = {"political": "r/politics", "world": "r/worldnews",
                       "pop_culture": "r/entertainment"}.get(ctx, "r/news")
        second = "r/OutOfTheLoop" if primary != "r/OutOfTheLoop" else "r/news"
        if primary in ("r/politics", "r/worldnews"):
            second = "r/news"
        return gt or primary, primary, [primary, second]
    return "r/memes", "r/memes", ["r/memes"]          # poor / irr / noise


def compute_karma(cat: str, num: int, virality: float) -> int:
    if cat == "edgy" and num == 1 and virality >= EDGY1_SPIKE_THRESHOLD:
        return EDGY1_SPIKE_VALUE
    low, high, nonlinear = KARMA_RANGES.get((cat, num), (400, 1200, False))
    v = virality ** 0.65 if nonlinear else virality
    return max(low, round((low + v * (high - low)) / 100) * 100)


def fmt_upvotes(n: int) -> str:
    if n >= 1000:
        k = n / 1000
        return f"{int(k)}K" if k == int(k) else f"{k:.1f}K"
    return str(n)


def load_news_articles() -> list[dict]:
    """99_170.txt holds 72 articles as (title, dateline, body) triples, in instance order 99..170."""
    lines = [l for l in NEWS_TXT.read_text(encoding="utf-8").split("\n") if l.strip()]
    arts = []
    i = 0
    while i + 2 < len(lines):
        title, dateline, body = lines[i], lines[i + 1], lines[i + 2]
        if not dateline.startswith("(Dateline"):
            raise RuntimeError(f"99_170.txt out of sync at line {i}: {dateline[:40]}")
        title = re.sub(r"^\d+\.\s*", "", title).strip()
        arts.append({"title": title, "dateline": dateline.strip("() ").replace("Dateline: ", ""), "body": body.strip()})
        i += 3
    return arts


def load_news_articles_2() -> dict[int, dict]:
    """News_Summary_171-230.txt: '<N>. <Title>' line followed by a body line. Returns {N: article}."""
    lines = [l for l in NEWS_TXT_2.read_text(encoding="utf-8-sig").split("\n") if l.strip()]
    arts, i = {}, 0
    while i < len(lines):
        m = re.match(r"^(\d+)\.\s*(.+)$", lines[i].strip())
        if not (m and i + 1 < len(lines)):
            raise RuntimeError(f"News_Summary_171-230.txt out of sync at: {lines[i][:60]}")
        arts[int(m.group(1))] = {"title": m.group(2).strip(), "dateline": None, "body": lines[i + 1].strip()}
        i += 2
    return arts


def _split_title(title: str) -> tuple[str, str | None]:
    """'Title (December 1. 2023)' → ('Title', 'December 1, 2023')."""
    m = re.match(r"^(.*?)\s*\(([A-Z][a-z]+ \d{1,2})[.,]\s*(\d{4})\)\s*$", title)
    return (m.group(1).strip(), f"{m.group(2)}, {m.group(3)}") if m else (title.strip(), None)


def load_news_articles_3() -> list[dict]:
    """hello.txt: 'N. Title (Date)' + body lines (1–68), then '## N. Title' + body (revive).
    Returns the articles in file order."""
    arts = []
    for line in NEWS_TXT_3.read_text(encoding="utf-8-sig").split("\n"):
        m = re.match(r"^(?:##\s*)?\d+\.\s+(.+)$", line.strip())
        if m:
            title, date = _split_title(m.group(1))
            arts.append({"title": title, "dateline": date, "body": ""})
        elif line.strip():
            arts[-1]["body"] = f"{arts[-1]['body']} {line.strip()}".strip()
    return arts


def load_article_file(path: Path) -> dict:
    """A news_article.txt as written by this script: title, optional '(Dateline: …)', body."""
    lines = [l.strip() for l in path.read_text(encoding="utf-8-sig").split("\n") if l.strip()]
    dateline = None
    if len(lines) > 1 and lines[1].startswith("(Dateline:"):
        dateline = lines.pop(1).strip("() ").replace("Dateline: ", "")
    return {"title": lines[0], "dateline": dateline, "body": " ".join(lines[1:])}


def load_news_fixes() -> dict[str, dict]:
    """news_fixes.txt: '### key', 'Title (Date)', body. Returns {key: article}."""
    fixes, key = {}, None
    for line in NEWS_FIXES.read_text(encoding="utf-8").split("\n"):
        if line.startswith("### "):
            key = line[4:].strip()
        elif line.startswith("#") or not line.strip() or key is None:
            continue
        elif key not in fixes:
            title, date = _split_title(line)
            fixes[key] = {"title": title, "dateline": date, "body": ""}
        else:
            fixes[key]["body"] = f"{fixes[key]['body']} {line.strip()}".strip()
    return fixes


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    config = json.loads(CONFIG_IN.read_text())
    NEU_TAXONOMY_SUBS.update(s[2:] for s in config["subreddit_taxonomy"]["neutral_subs"]
                             + config["subreddit_taxonomy"]["niche_subs"])
    existing_keys = set(config["instances"])
    existing_nums = {int(re.match(r"^(\d+)_", d.name).group(1)) for d in list_dirs(DATASET_DIR)}
    assert set(range(1, 99)) <= existing_nums, "original 1–98 folders missing"   # re-runs rebuild 99+

    # megathread index
    mega_files = {f.name: f for d in MEGA_DIRS for f in d.iterdir()
                  if f.suffix == ".json" and not f.name.startswith("._")}
    mega_norm = {n: norm_match(n) for n in mega_files}

    # edgy fixes: same folder name as the raw folder, holding all memes incl. the new edgy one
    edgy_fix = {slugify(d.name): d for d in list_dirs(EDGY_FIX)}
    used_fix = set()

    # 1–2. enumerate + number
    plan, unordered = [], []
    for batch, mode, domain in BATCHES:
        for d in list_dirs(RAW / "instances" / batch):
            if slugify(d.name) in edgy_fix:
                used_fix.add(slugify(d.name))
                d = edgy_fix[slugify(d.name)]
            if mode == "prefixed":
                plan.append((int(re.match(r"^(\d+)_", d.name).group(1)), d, batch, domain))
            else:
                slug = SLUG_OVERRIDES.get(slugify(d.name), slugify(d.name))
                if slug in EXCLUDE_SLUGS:
                    print(f"  EXCLUDED: {d.name!r}"); continue
                if slug in ORDER_171_230:
                    plan.append((171 + ORDER_171_230.index(slug), d, batch, domain))
                else:
                    unordered.append((d, batch, domain))
    assert len({t[0] for t in plan}) == len(plan), "duplicate numbers"
    assert len([t for t in plan if t[0] >= 171]) == len(ORDER_171_230), "ORDER_171_230 slug not found in raw folders"
    assert not unordered, f"raw folders not in ORDER_171_230: {[t[0].name for t in unordered]}"
    assert used_fix == set(edgy_fix), f"edgy-fix folders with no raw folder: {set(edgy_fix) - used_fix}"
    plan.sort(key=lambda t: t[0])
    nums = [p[0] for p in plan]
    assert nums == list(range(99, 231)), "numbering gap"
    print(f"{len(plan)} instances from MemeECON/instances → #{nums[0]}–#{nums[-1]} "
          f"({len(used_fix)} with an edgy fix)")

    articles = load_news_articles()
    assert len(articles) == 72, len(articles)
    articles2 = load_news_articles_2()
    assert sorted(articles2) == list(range(171, 231)), "expected articles 171–230"
    articles3 = load_news_articles_3()
    assert len(articles3) == len(ORDER_231_300) == 70, (len(articles3), len(ORDER_231_300))
    fixes = load_news_fixes()
    assert set(fixes) <= {o[0] for o in ORDER_231_300}, set(fixes) - {o[0] for o in ORDER_231_300}
    for (key, *_, kw), art in zip(ORDER_231_300, articles3):
        assert kw.lower() in art["title"].lower(), f"hello.txt out of order: {key} ↔ {art['title']!r}"

    # pass 1: resolve everything, collect post scores for virality normalisation
    resolved = []
    used_mega = set()
    for num, d, batch, domain in plan:
        slug = slugify(d.name)
        slug = SLUG_OVERRIDES.get(slug, slug)
        assert slug and slug not in existing_keys, (d.name, slug)
        folder_name = f"{num}_{slug}"

        # megathread
        if slug in MEGA_OVERRIDES:
            mname = MEGA_OVERRIDES[slug]
        else:
            n = norm_match(d.name); best, bs = None, 0.0
            for mn, mnorm in mega_norm.items():
                ta, tb = set(n.split()), set(mnorm.split())
                jac = len(ta & tb) / max(1, len(ta | tb))
                sc = max(jac, difflib.SequenceMatcher(None, n, mnorm).ratio())
                if sc > bs:
                    bs, best = sc, mn
            mname = best if bs >= 0.5 else None
        if mname:
            assert mname in mega_files, mname
            assert mname not in used_mega, f"megathread {mname} claimed twice"
            used_mega.add(mname)
            mega = json.loads(mega_files[mname].read_text())
            score = int(mega.get("post", {}).get("score", 0) or 0)
            neu_sub = mega.get("post", {}).get("subreddit") or None
        else:
            mega, score, neu_sub = None, 0, None

        memes = normalise_memes(d, slugify(d.name))
        assert memes, f"no memes in {d}"
        article = articles[num - 99] if 99 <= num <= 170 else articles2.get(num)
        ctx = context_for(slug, domain)
        resolved.append(dict(num=num, raw=d, slug=slug, folder=folder_name, batch=batch, domain=domain,
                             ctx=ctx, mega_name=mname, mega=mega, score=score, neu_sub=neu_sub,
                             memes=memes, article=article, news_source=None))

    unused = sorted(set(mega_files) - used_mega)
    print("unmatched megathreads:", unused)

    # 231–300: explicit order, folders and megathreads (raw names are not descriptive enough to match)
    used_mega3 = set()
    for k, ((key, src, mname, domain, ctx, _kw), art) in enumerate(zip(ORDER_231_300, articles3)):
        assert key not in existing_keys and key not in {r["slug"] for r in resolved}, key
        assert src.is_dir(), src
        if mname:
            assert (MEGA_DIR_3 / mname).is_file() and mname not in used_mega3, mname
            used_mega3.add(mname)
        article, news_source = fixes.get(key, art), ("news_fixes.txt" if key in fixes else "hello.txt")
        hand_off = MISSING_DIR / f"{231 + k}_{key}"
        if hand_off.is_dir():
            # raw folder names drive filename prefixes (173scp…), so keep the original for parsing
            raw_slug = slugify(src.name)
            src, mpath = hand_off, hand_off / "megathread.json"
            mname = str(mpath.relative_to(ROOT)) if mpath.is_file() else None
            if (hand_off / "news_article.txt").is_file():
                article, news_source = load_article_file(hand_off / "news_article.txt"), "19_missing"
            elif key in REPLACED_EVENTS:
                # the hello.txt article describes the old event, not this one
                article, news_source = None, None
        else:
            raw_slug, mpath = slugify(src.name), (MEGA_DIR_3 / mname if mname else None)
        if mname:
            mega = json.loads(mpath.read_text())
            score = int(mega.get("post", {}).get("score", 0) or 0)
            neu_sub = mega.get("post", {}).get("subreddit") or None
        else:
            mega, score, neu_sub = None, 0, None
        resolved.append(dict(num=231 + k, raw=src, slug=key, folder=f"{231 + k}_{key}", batch="part3",
                             domain=domain, ctx=ctx, mega_name=mname, mega=mega, score=score, neu_sub=neu_sub,
                             memes=normalise_memes(src, raw_slug, key), hand_off=hand_off.is_dir(),
                             article=article, news_source=news_source))
    unused3 = sorted(f.name for f in MEGA_DIR_3.glob("*.json")
                     if not f.name.startswith("._") and f.name not in used_mega3)
    print("unmatched megathreads in Update/megathread:", unused3)
    print("instances without megathread:", [r["folder"] for r in resolved if not r["mega"]])

    # virality is normalised by the 99–230 maximum so adding instances never changes their karma
    max_log = math.log(max(r["score"] for r in resolved if r["num"] <= 230) + 1)

    if not dry:
        for d in list_dirs(DATASET_DIR):
            if int(re.match(r"^(\d+)_", d.name).group(1)) >= 99:
                shutil.rmtree(d, ignore_errors=True)   # derived; AppleDouble ._ files vanish mid-walk
                assert not d.exists(), d

    # pass 2: build folders + config entries
    new_cfg = {}
    manifest = []
    for r in resolved:
        virality = min(1.0, math.log(r["score"] + 1) / max_log)
        dest = DATASET_DIR / r["folder"]
        karma_lines, memes_cfg, relevant = [], {}, {}
        for new_name, cat, n, src, old_name in r["memes"]:
            upv = compute_karma(cat, n, virality)
            gt_sub, primary, all_fit = subreddits_for(cat, r["ctx"], r["neu_sub"])
            karma_lines.append(f"{new_name} - {fmt_upvotes(upv)} upvotes subreddit {gt_sub}")
            tier, tier_label = TIER[cat]
            memes_cfg[f"{cat}{n}"] = {
                "file": new_name, "category": cat, "tier": tier, "tier_label": tier_label, "number": n,
                "base_karma": None, "karma_source": "generated_log_virality",
                "primary_subreddit": primary, "all_fit_subreddits": all_fit,
                "gt_upvotes": upv, "gt_subreddit": gt_sub,
            }
            for s in all_fit:
                relevant.setdefault(s, cat)
        if memes_cfg:
            top = max(m["gt_upvotes"] for m in memes_cfg.values())
            for m in memes_cfg.values():
                m["base_karma"] = round(m["gt_upvotes"] / top * 1000)

        used_src = {src.name for _, _, _, src, _ in r["memes"]}
        unlabelled = [f.name for f in list_images(r["raw"])
                      if f.name not in used_src and (r["slug"], f.name) not in IGNORE_FILES]
        issues = []
        if not memes_cfg:
            issues.append("memes unlabelled")
        elif not any(m["category"] == "edgy" for m in memes_cfg.values()):
            issues.append("no edgy (Tier 5) meme")
        if 0 < len(memes_cfg) < 4:
            issues.append(f"only {len(memes_cfg)} memes")
        if unlabelled:
            issues.append(f"unlabelled files: {', '.join(unlabelled)}")
        if not r["mega"]:
            issues.append("no megathread")
        if not r["article"]:
            issues.append("no news article")
        if r.get("hand_off"):
            msg, still = BLOCKER_CHECKS.get(r["slug"], (None, None))
            if still and still(r["raw"]):
                issues.append(msg)
        elif r["slug"] in BLOCKERS:
            issues.append(BLOCKERS[r["slug"]])

        entry = {
            "split": "pending" if issues else "extension",
            "folder": r["folder"],
            "has_megathread": r["mega"] is not None,
            "megathread_file": "dis_mega.json" if r["mega"] else None,
            "has_webpage": False, "webpage_file": None,
            "has_news_text": r["article"] is not None,
            "news_text_file": "news_article.txt" if r["article"] else None,
            "has_karma_gt": True,
            "meme_count": len(memes_cfg),
            "has_edgy": any(m["category"] == "edgy" for m in memes_cfg.values()),
            "domain": r["domain"], "context": r["ctx"], "source_batch": r["batch"],
            "megathread_post_score": r["score"], "virality": round(virality, 4),
            "memes": dict(sorted(memes_cfg.items())),
            "relevant_subreddits": relevant,
        }
        if issues:
            entry["issues"] = issues
        if r["slug"] in WARNINGS:
            entry["warnings"] = [WARNINGS[r["slug"]]]
        if r["news_source"] == "news_fixes.txt":
            entry["news_source"] = "news_fixes.txt (drafted, replaces the hello.txt article about a different event)"
        new_cfg[r["slug"]] = entry
        raw_rel = r["raw"].relative_to(RAW) if r["raw"].is_relative_to(RAW) else r["raw"].relative_to(ROOT)
        manifest.append({
            "num": r["num"], "key": r["slug"], "folder": r["folder"], "raw_folder": str(raw_rel),
            "megathread_source": r["mega_name"], "news_source": r["news_source"],
            "domain": r["domain"], "context": r["ctx"], "split": entry["split"],
            "has_edgy": entry["has_edgy"], "has_news_text": entry["has_news_text"],
            "file_renames": {old: new for new, _, _, _, old in r["memes"] if old != new},
            "issues": issues, "warnings": entry.get("warnings", []),
        })

        flag = "" if not issues else f"   [PENDING: {'; '.join(issues)[:90]}]"
        print(f"  #{r['num']:3d} {r['folder'][:55]:<55} {r['ctx']:<11} v={virality:.2f} "
              f"memes={len(memes_cfg)} mega={'y' if r['mega'] else 'N'} news={'y' if r['article'] else '-'}{flag}")

        if dry:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for new_name, cat, n, src, old_name in r["memes"]:
            shutil.copy2(src, dest / new_name)
        if r["mega"]:
            (dest / "dis_mega.json").write_text(json.dumps(r["mega"], ensure_ascii=False, indent=1))
        if r["article"]:
            a = r["article"]
            dateline = f"(Dateline: {a['dateline']})\n" if a["dateline"] else ""
            (dest / "news_article.txt").write_text(f"{a['title']}\n{dateline}{a['body']}\n", encoding="utf-8")
        (dest / "karma_count.txt").write_text("\n".join(karma_lines) + "\n")

    # 7. merged config
    merged = json.loads(CONFIG_IN.read_text())
    merged["_meta"] = dict(merged["_meta"], version="3.0",
                           generated_by="finalize_dataset.py (1–98) + extend_dataset.py (99–300)",
                           extension_note="Instances 99–300 use log-virality karma from megathread post.score "
                                          "(karma_source=generated_log_virality). domain=pop_culture marks "
                                          "film/TV/internet-culture instances. split: paper98 = the paper's "
                                          "instances, extension = complete new instances, pending = incomplete "
                                          "(see 'issues'; do not run).")
    for v in merged["instances"].values():
        v.setdefault("split", "paper98")
    merged["instances"].update(new_cfg)
    print(f"\nconfig: {len(config['instances'])} original + {len(new_cfg)} new = {len(merged['instances'])}")
    pending = [k for k, v in new_cfg.items() if v["split"] == "pending"]
    print(f"extension (complete): {len(new_cfg) - len(pending)}   pending ({len(pending)}): {pending}")
    if not dry:
        CONFIG_OUT.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"wrote {CONFIG_OUT.name}, {MANIFEST.relative_to(ROOT)}, {len(new_cfg)} folders in {DATASET_DIR.name}/")
    else:
        print("[dry-run] nothing written")


if __name__ == "__main__":
    main()
