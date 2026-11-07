#!/usr/bin/env python3
"""
generate_news_pages.py
======================
Builds the Wikipedia-styled news page (<safe>_news.html) for every instance.

Single source of truth: <instance folder>/news_article.txt
    line 1          title
    line 2 (opt.)   "(Dateline: <date>)"
    rest            body; paragraphs separated by blank lines
                    (a single-line body is auto-split into paragraphs)

Each page's "Go to Reddit Thread" button links to the instance's megathread, read
from memeeconomy_tasks.json, so run this after environment/inject_instances.py.
The pages are served at http://localhost:9999/test_pages/<safe>_news.html and
/news/<safe>_news.html (environment/setup_environment.py copies them in).

Usage:
    python environment/generate_news_pages.py                 # → generated_pages/
    python environment/generate_news_pages.py --config configs/instance_config_20pct.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "memes"
TASKS_PATH  = ROOT / "memeeconomy_tasks.json"
OUTPUT_DIR  = ROOT / "generated_pages"
SECTIONS    = ["Background", "Details", "Public Reaction", "Cultural Impact", "Aftermath", "Legacy"]


def safe_name(instance_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", instance_name)


# ── article text ─────────────────────────────────────────────────────────────
def split_paragraphs(text: str, per: int = 4, cap: int = 20) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'“])", text.strip())
    if len(sents) > per * cap:
        per = len(sents) // cap + 1
    paras, chunk = [], []
    for s in sents:
        chunk.append(s)
        if len(chunk) == per:
            paras.append(" ".join(chunk)); chunk = []
    if chunk:
        paras.append(" ".join(chunk))
    return paras[:cap] if paras else [text.strip()]


def read_article(folder: Path) -> tuple[str, str | None, list[str]] | None:
    f = folder / "news_article.txt"
    if not f.exists():
        return None
    lines = f.read_text(encoding="utf-8").split("\n")
    title = lines[0].strip()
    rest = lines[1:]
    dateline = None
    if rest and rest[0].strip().startswith("(Dateline"):
        dateline = rest[0].strip(); rest = rest[1:]
    body = "\n".join(rest).strip()
    paras = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", body) if p.strip()]
    if len(paras) == 1:
        paras = split_paragraphs(paras[0])
    if dateline:
        paras[0] = f"{dateline} {paras[0]}"
    return title, dateline, paras


# ── HTML ─────────────────────────────────────────────────────────────────────
CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Linux Libertine',Georgia,Times,serif;font-size:14px;line-height:1.6;color:#202122;background:#f8f9fa}
.wh{background:#fff;border-bottom:1px solid #a7d7f9;padding:8px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:100;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.wl{font-size:21px;font-weight:bold;white-space:nowrap}.wl span{color:#3366cc}
.ws{flex:1;max-width:380px;display:flex;border:1px solid #a2a9b1;border-radius:2px;overflow:hidden}
.ws input{flex:1;padding:6px 8px;border:none;outline:none;font-size:13px}
.ws button{background:#f8f9fa;border:none;border-left:1px solid #a2a9b1;padding:6px 10px;cursor:pointer}
.wn{display:flex;gap:12px;font-size:12px;margin-left:auto}.wn a{color:#3366cc;text-decoration:none}
.wb{max-width:960px;margin:0 auto;padding:20px 20px 60px;background:#fff;min-height:100vh}
.wt{border-bottom:1px solid #a2a9b1;margin-bottom:16px;display:flex}
.tab{padding:6px 14px;font-size:13px;color:#3366cc;cursor:pointer;border:1px solid transparent;border-bottom:none;margin-bottom:-1px}
.tab.a{background:#fff;border-color:#a2a9b1;color:#202122;border-radius:2px 2px 0 0}
h1.at{font-family:'Linux Libertine',Georgia,Times,serif;font-size:28px;font-weight:normal;border-bottom:1px solid #a2a9b1;padding-bottom:4px;margin-bottom:10px}
.am{font-size:12px;color:#54595d;margin-bottom:14px;font-style:italic}
.ib{float:right;clear:right;margin:0 0 14px 20px;background:#f8f9fa;border:1px solid #a2a9b1;font-size:12.5px;width:250px}
.ibt{background:#cee0f2;text-align:center;font-weight:bold;padding:5px 8px;font-size:13px}
.ibr{display:flex;border-top:1px solid #a2a9b1;padding:4px 8px;gap:8px}
.ibl{font-weight:bold;min-width:75px}
.toc{background:#f8f9fa;border:1px solid #a2a9b1;padding:10px 14px;display:inline-block;min-width:190px;margin-bottom:16px;font-size:13px}
.toc b{display:block;margin-bottom:5px}
.toc ol{padding-left:18px}.toc li{margin:2px 0}.toc a{color:#3366cc;text-decoration:none}
.ab p{margin-bottom:13px;text-align:justify}
h2.sh{font-family:'Linux Libertine',Georgia,Times,serif;font-size:20px;font-weight:normal;border-bottom:1px solid #a2a9b1;padding-bottom:3px;margin:20px 0 9px}
.wf{border-top:1px solid #a2a9b1;margin-top:38px;padding-top:12px;font-size:12px;color:#54595d;text-align:center}
.wf a{color:#3366cc;text-decoration:none}
.rb{position:fixed;top:60px;right:18px;background:#FF4500;color:#fff;padding:9px 16px;text-decoration:none;border-radius:5px;font-family:sans-serif;font-weight:bold;font-size:13px;z-index:999;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.rb:hover{background:#e03d00}"""


def build_html(title: str, paragraphs: list[str], reddit_url: str) -> str:
    body, si = "", 0
    for i, p in enumerate(paragraphs):
        if i > 0 and i % 3 == 0 and si < len(SECTIONS):
            body += f'<h2 class="sh">{SECTIONS[si]}</h2>\n'; si += 1
        body += f"<p>{html.escape(p, quote=False)}</p>\n"
    toc = "".join(f"<li><a href='#'>{j + 1} {s}</a></li>" for j, s in enumerate(SECTIONS[:max(si, 1)]))
    wc = sum(len(p.split()) for p in paragraphs)
    rt = max(1, wc // 200)
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t} - Wikipedia</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="wh">
  <div class="wl">&#9776; <span>Wikipedia</span></div>
  <div class="ws"><input type="text" placeholder="Search Wikipedia" value="{t[:45]}"><button>&#128269;</button></div>
  <div class="wn"><a href="#">Create account</a><a href="#">Log in</a><a href="#">Donate</a></div>
</div>

<a href="{reddit_url}" class="rb">Go to Reddit Thread &rarr;</a>

<div class="wb">
  <div class="wt">
    <div class="tab a">Article</div><div class="tab">Talk</div>
    <div class="tab" style="margin-left:auto">Read</div><div class="tab">Edit</div><div class="tab">View history</div>
  </div>

  <h1 class="at">{t}</h1>
  <div class="am">From Wikipedia, the free encyclopedia &nbsp;|&nbsp; ~{wc} words &nbsp;|&nbsp; {rt} min read</div>

  <div class="ib">
    <div class="ibt">Event Overview</div>
    <div class="ibr"><span class="ibl">Topic</span><span>Internet Culture / Current Events</span></div>
    <div class="ibr"><span class="ibl">Coverage</span><span>Global / Social Media</span></div>
    <div class="ibr"><span class="ibl">Impact</span><span>High &mdash; Viral Meme Event</span></div>
    <div class="ibr"><span class="ibl">Platforms</span><span>Reddit, X, TikTok, Instagram</span></div>
  </div>

  <div class="toc"><b>Contents</b><ol>{toc}</ol></div>

  <div class="ab" style="clear:both">{body}</div>

  <div class="wf">
    <p>This page was last edited recently. &nbsp;|&nbsp;
       <a href="#">Privacy policy</a> &nbsp;|&nbsp;
       <a href="#">About Wikipedia</a> &nbsp;|&nbsp;
       <a href="#">Disclaimers</a></p>
    <p style="margin-top:7px">Text available under <a href="#">Creative Commons Attribution-ShareAlike License 4.0</a>.</p>
  </div>
</div>
</body>
</html>"""


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/instance_config.json")
    args = ap.parse_args()

    config = json.loads((ROOT / args.config).read_text())

    tasks = {}
    if TASKS_PATH.exists():
        tasks = json.loads(TASKS_PATH.read_text()).get("injected_tasks", {})

    OUTPUT_DIR.mkdir(exist_ok=True)
    built, missing = 0, []
    for name, data in config["instances"].items():
        art = read_article(DATASET_DIR / data["split"] / data["folder"])
        if art is None:
            missing.append(name); continue
        title, _, paras = art
        url = tasks.get(name, {}).get("url", "#")
        (OUTPUT_DIR / f"{safe_name(name)}_news.html").write_text(build_html(title, paras, url), encoding="utf-8")
        built += 1
    print(f"built {built} news pages → {OUTPUT_DIR.name}/ ; no news_article.txt: {missing}")
    no_url = [n for n in config["instances"] if n not in tasks]
    print(f"instances without an injected Reddit URL yet (button → '#'): {len(no_url)}")

    issues = []
    for name in config["instances"]:
        p = OUTPUT_DIR / f"{safe_name(name)}_news.html"
        if not p.exists():
            continue
        page = p.read_text(encoding="utf-8")
        for needle, label in [("<!DOCTYPE html>", "no doctype"), ("Go to Reddit Thread", "no reddit button"),
                              ('class="at"', "no title"), ("<p>", "no paragraphs")]:
            if needle not in page:
                issues.append(f"{name}: {label}")
    print("structure check:", "OK" if not issues else issues)


if __name__ == "__main__":
    main()
