#!/usr/bin/env python3
"""
memeeconomy/run_single_agent.py
=============================
MemeEconomy single-agent benchmark runner.

By default only the 98 instances used in the paper (split "paper98") are played.
Pass --split extension or --split all to include instances added after the paper.

Usage examples
--------------
# Single model, 10 rounds, normal pressure
python -m memeeconomy.run_single_agent --models qwen32b --rounds 10 --condition normal

# Multiple models, desperation pressure, visible browser
python -m memeeconomy.run_single_agent --models gpt5_1 gemini2_5_pro --rounds 20 --condition desperation --visible

# Resume a crashed run
python -m memeeconomy.run_single_agent --models gpt5_1 --rounds 23 --resume-portfolio results/gpt5_1/portfolio_run4_20260320_031844.json

# Human baseline
python -m memeeconomy.run_single_agent --human --rounds 10
"""

import json
import time
import base64
import re
import argparse
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os

from memeeconomy.portfolio_manager import PortfolioManager
from memeeconomy.prompts import (
    build_system_prompt,
    build_user_message,
    extract_decision,
    CONDITIONS,
    get_condition_metadata,
)
from memeeconomy.prompts_react import (
    build_react_system_prompt,
    build_react_user_message,
    parse_react_decision,
    react_to_bdi_raw,
)
from memeeconomy.prompts_nobdi import (
    build_nobdi_system_prompt,
    build_nobdi_user_message,
    nobdi_to_bdi_raw,
)
from memeeconomy.verifier_agent import VerifierAgent

# ── Configuration ─────────────────────────────────────────────────────────────

def _load_openrouter_key() -> str:
    """OPENROUTER_KEY from the environment, else from a .env file next to this script."""
    key = os.environ.get("OPENROUTER_KEY", "")
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == "OPENROUTER_KEY":
                key = v.strip().strip('"').strip("'")
    return key


OPENROUTER_KEY = _load_openrouter_key()
SPLITS          = ("paper98", "extension", "all")
BASE_URL       = "http://localhost:9999"
MAX_STEPS       = 25          # hard cap per round
LLM_TIMEOUT     = 120         # seconds per API call
LLM_RETRIES     = 3
TEMPERATURE     = 0.0         # deterministic output

# Minimum scrolls per page type before agent may decide.
# These are soft: we log violations but do NOT override the agent's action.
MIN_SCROLLS = {
    "news":      2,
    "reddit":    3,
    "selection": 2,
}

MODELS: Dict[str, str] = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt5_1":           "openai/gpt-5.1",
    # "gpt4o":          "openai/gpt-4o-2024-11-20",

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude_haiku5_5":  "anthropic/claude-haiku-5.5",
    "claude_sonnet5_5": "anthropic/claude-sonnet-5.5",
    "claude_opus5_5":   "anthropic/claude-opus-5.5",

    # ── Google ────────────────────────────────────────────────────────────────
    "gemini3_flash":   "google/gemini-3-flash-preview",
    "gemini3_1_pro":    "google/gemini-3.1-pro-preview",
    "gemini2_5_pro":    "google/gemini-2.5-pro",
    "gemma3_27B":       "google/gemma-3-27b-it",
    "gemma3_4B":        "google/gemma-3-4b-it",
    "gemma4_31B":       "google/gemma-4-31b-it",

    # ── Qwen (multimodal VL) ──────────────────────────────────────────────────
    "qwen235b":         "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen32b":          "qwen/qwen3-vl-32b-instruct",
    "qwen8b":           "qwen/qwen3-vl-8b-instruct",
    "qwen3_VL_8B":      "qwen/qwen3-vl-8b-instruct",

    # ── Qwen (text-only / flash) ──────────────────────────────────────────────
    "qwen3_5_9B":       "qwen/qwen3.5-9b",
    "qwen3_14B":        "qwen/qwen3-14b",
    "qwen3_5_flash":    "qwen/qwen3.5-flash-02-23",
    "qwen3_5_27b":      "qwen/qwen3.5-27b",
    "qwen3_vl_8b_think":"qwen/qwen3-vl-8b-thinking",
    "qwen3_6_plus":     "qwen/qwen3.6-plus",

    # ── Meta Llama ────────────────────────────────────────────────────────────
    "llama4_maverick":  "meta-llama/llama-4-maverick",
    "llama3_2_11b":     "meta-llama/llama-3.2-11b-vision-instruct",

    # ── InternVL ──────────────────────────────────────────────────────────────
    "internvl3_78b":    "opengvlab/internvl3-78b",
    "internvl3_14b":    "opengvlab/internvl3-14b",

    # ── Google Gemma (additional sizes) ───────────────────────────────────────
    "gemma3_12B":       "google/gemma-3-12b-it",

    # ── Mistral / Ministral ───────────────────────────────────────────────────
    "mistral_large":    "mistralai/mistral-large-2512",
    "mistral_small_4":  "mistralai/mistral-small-2603",
    "mistral_small_3_2":"mistralai/mistral-small-3.2-24b-instruct",
    "ministral_14":     "mistralai/ministral-14b-2512",
    "ministral_14B":    "mistralai/ministral-14b-2512",
    "ministral_8B":     "mistralai/ministral-8b-2512",
    "ministral_3B":     "mistralai/ministral-3b-2512",
}

BASE_DIR    = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "instance_config.json"
TASKS_PATH  = BASE_DIR / "memeeconomy_tasks.json"
RESULTS_DIR = BASE_DIR / "results"
PAGES_DIR   = BASE_DIR / "generated_pages"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)


def available_instances(tasks_data: dict, split: str = "paper98") -> List[str]:
    """Injected instances in the given split, in config (= dataset folder) order."""
    with open(CONFIG_PATH, "r") as f:
        config_instances = json.load(f).get("instances", {})
    return [
        k for k, v in config_instances.items()
        if (split == "all" or v.get("split") == split)
        and tasks_data.get(k, {}).get("url", "#") != "#"
    ]


def _next_run_id(model_dir: Path) -> int:
    existing = list(model_dir.glob("portfolio_run*.json"))
    if not existing:
        return 1
    nums = [int(m.group(1)) for p in existing
            for m in [re.search(r"portfolio_run(\d+)_", p.name)] if m]
    return max(nums) + 1 if nums else 1


def _stable_round_seed(run_seed: int, round_index: int, instance_name: str) -> int:
    """Common random number for matched conditions, stable across processes."""
    payload = f"{run_seed}\0{round_index}\0{instance_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _page_type(url: str) -> str:
    if "portfolio_dashboard" in url:
        return "dashboard"
    if "_news" in url:
        return "news"
    if "/f/" in url:
        return "reddit"
    if "selection_" in url:
        return "selection"
    return "unknown"


def _count_scrolls_since_last_click(trajectory: List[dict]) -> int:
    count = 0
    for act in reversed(trajectory):
        if act.get("action") == "scroll":
            count += 1
        elif act.get("action") == "click":
            break
    return count


# ── Subreddit / meme list builders ────────────────────────────────────────────

def _build_subreddit_list(taxonomy: dict, instance_name: str = "") -> List[dict]:
    """
    Return a capped, deterministic subreddit list (~10-12 entries).
    Seeded by instance_name so every run of the same instance sees the same list,
    but different instances get different samples.
    """
    import random as _r
    rng = _r.Random(instance_name)  # deterministic per instance

    ban_risk = set(taxonomy.get("meme_ban_risk_subs", []))
    type_label = {
        "meme_subs":           "General meme",
        "political_meme_subs": "Political meme",
        "neutral_subs":        "News & discussion",
        "niche_subs":          "Niche community",
    }

    subs = []
    for stype, slist in taxonomy.items():
        if stype == "meme_ban_risk_subs" or not isinstance(slist, list):
            continue
        label   = type_label.get(stype, "Community")
        # Take at most 3 from each category
        sample  = rng.sample(slist, min(3, len(slist)))
        for sub in sample:
            subs.append({
                "name":          sub,
                "description":   label,
                "meme_ban_risk": sub in ban_risk,
            })

    rng.shuffle(subs)
    return subs[:12]


def _build_meme_list(instance_name: str, instance_config_data: dict) -> List[dict]:
    """
    Return memes sorted by meme_id so the letter assignment is deterministic
    across runs, OS, and Python versions.
    """
    safe = _safe_name(instance_name)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    memes_sorted = sorted(instance_config_data.get("memes", {}).items())
    return [
        {
            "id":        m_id,
            "letter":    letters[idx % 26],
            "tier":      m_dat.get("tier", 1),
            "image_url": f"http://localhost:9999/newsfeed/{safe}/{m_dat.get('file')}",
        }
        for idx, (m_id, m_dat) in enumerate(memes_sorted)
    ]


# ── Page renderer ─────────────────────────────────────────────────────────────

def _render_round_pages(
    pm, idx: int, num_rounds: int, instance_name: str,
    model_id: str, env, tasks_data: dict,
    taxonomy: dict, instance_config_data: dict,
    run_label: str = "",
) -> tuple:
    """Render dashboard + selection pages and sync to Docker container."""
    safe = _safe_name(instance_name)
    dd   = pm.get_dashboard_data()

    db_html = env.get_template("portfolio_dashboard_gemini.html").render(
        agent_id            = dd["agent_id"],
        current_round       = dd["current_round"] + 1,
        total_rounds        = dd["total_rounds"],
        rounds_remaining    = dd["rounds_remaining"] - 1,
        is_complete         = dd["is_complete"],
        karma_balance       = dd["karma"]["balance"],
        karma_return_pct    = dd["karma"]["return_pct"],
        win_rate            = dd["performance"]["win_rate"],
        win_count           = dd["performance"]["win_count"],
        loss_count          = dd["performance"]["loss_count"],
        current_streak      = dd["performance"]["current_streak"],
        best_round_karma    = dd["performance"]["best_round"]["karma"],
        best_round_num      = dd["performance"]["best_round"]["round"],
        worst_round_karma   = dd["performance"]["worst_round"]["karma"],
        worst_round_num     = dd["performance"]["worst_round"]["round"],
        avg_karma_per_round = dd["performance"]["avg_karma_per_round"],
        ethical_violations  = dd["behavioral"]["ethical_violations"],
        tier5_selections    = dd["behavioral"]["tier5_selections"],
        tier5_rate          = dd["behavioral"]["tier5_rate"],
        risk_score          = dd["behavioral"]["risk_score"],
        removals            = dd["removals"]["total"],
        tier_distribution   = dd["behavioral"]["tier_distribution"],
        total_picks         = dd["current_round"],
        unique_subs         = dd["behavioral"]["unique_subs_used"],
        perfect_fits        = dd["behavioral"]["perfect_fit_count"],
        wrong_market        = dd["behavioral"]["wrong_market_count"],
        round_history       = dd["round_history"],
        pressure_condition  = dd["pressure_condition"],
        session_start       = dd["session_start"],
        next_round_url      = f"{instance_name.replace(' ','_').replace('/','_')}_news.html",
    )
    with open(PAGES_DIR / f"portfolio_dashboard_{model_id}{run_label}.html", "w") as f:
        f.write(db_html)

    all_subs = _build_subreddit_list(taxonomy, instance_name)
    memes    = _build_meme_list(instance_name, instance_config_data)

    sel_html = env.get_template("meme_selection_gemini.html").render(
        round_number    = idx,
        total_rounds    = num_rounds,
        karma_balance   = pm.karma_balance,
        event_name      = f"Trending Topic #{idx}",
        event_description = "Research the context from the provided links",
        webpage_url     = f"http://localhost:9999/test_pages/{safe}_news.html",
        megathread_url  = tasks_data.get(instance_name, {}).get("url", "#"),
        memes           = memes,
        subreddits      = all_subs,
    )
    sel_filename = f"selection_{model_id}{run_label}_{instance_name.replace(' ','_').replace('/','_')}_gemini.html"
    with open(PAGES_DIR / sel_filename, "w") as f:
        f.write(sel_html)

    # Copy only this round's two pages. Copying the whole of generated_pages/ makes
    # files briefly unreadable (mode 700 until chmod), which serves nginx 403s to any
    # other run sharing the container.
    dest = "/var/www/html/public/test_pages/"
    round_pages = [f"portfolio_dashboard_{model_id}{run_label}.html", sel_filename]
    for page in round_pages:
        subprocess.run(["docker", "cp", str(PAGES_DIR / page), f"reddit_benchmark:{dest}"], check=False)
    subprocess.run(
        ["docker", "exec", "reddit_benchmark", "chmod", "755", *[dest + p for p in round_pages]],
        check=False,
    )
    return memes, all_subs


# ── Agent ─────────────────────────────────────────────────────────────────────

class MemeAgent:
    def __init__(self, model_key: str, model_dir: Path, condition: str, run_id: int = 1,
                 target_karma: Optional[int] = None, arch: str = "bdi",
                 run_seed: Optional[int] = None,
                 harm_assessment: bool = False):
        import requests
        self.client       = requests
        self.model_key    = model_key
        self.model        = MODELS.get(model_key, model_key)
        self.api_key      = OPENROUTER_KEY
        self.results_dir  = model_dir
        self.run_id       = run_id
        self.condition    = condition
        self.target_karma = target_karma
        self.arch         = arch   # "bdi" | "react"
        self.run_seed     = run_seed
        self.harm_assessment = harm_assessment

        self.pw          = None
        self.browser     = None
        self.page        = None
        self._owns_pw    = True   # overridden by start_browser(pw=...)
        self.screenshots_dir = model_dir / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

    # ── Browser ───────────────────────────────────────────────────────────────

    def start_browser(self, headless: bool = True, pw=None):
        """
        Start a Playwright browser.

        Parameters
        ----------
        pw : optional shared Playwright instance.
             When provided (multi-agent use), this agent does NOT own it
             and will not stop it in stop_browser().
             When None, a fresh sync_playwright() is started and owned here.
        """
        from playwright.sync_api import sync_playwright
        if pw is not None:
            self.pw          = pw
            self._owns_pw    = False
        else:
            self.pw          = sync_playwright().start()
            self._owns_pw    = True

        self.browser = self.pw.chromium.launch(
            headless=headless,
            slow_mo=100,
            channel="chrome",
        )
        self.page    = self.browser.new_page()
        self.page.set_viewport_size({"width": 1400, "height": 900})

    def stop_browser(self):
        try:
            if self.browser:
                self.browser.close()
            if self._owns_pw and self.pw:
                self.pw.stop()
        except Exception:
            pass

    def login(self) -> bool:
        try:
            self.page.goto(f"{BASE_URL}/login", timeout=30_000)
            time.sleep(2)
            if self.page.locator("input[name='_username']").count() > 0:
                self.page.fill("input[name='_username']", "MarvelsGrantMan136")
                self.page.fill("input[name='_password']", "test1234")
                self.page.click("button[type='submit']")
                self.page.wait_for_load_state("networkidle", timeout=15_000)
            return True
        except Exception as e:
            print(f"  [Login] {e}")
            return False

    def screenshot(self, prefix: str, meme_selection_url: str = "") -> str:
        try:
            url = self.page.url
            if "/f/" in url and meme_selection_url:
                self.page.evaluate(f"""
                    if (!document.getElementById('ai-proceed-btn')) {{
                        let btn = document.createElement('a');
                        btn.id = 'ai-proceed-btn';
                        btn.href = '{meme_selection_url}';
                        btn.innerText = 'Proceed to Meme Selection →';
                        btn.style.cssText = 'position:fixed;top:10px;right:10px;background:#28a745;color:white;padding:15px 30px;font-size:20px;font-weight:bold;z-index:2147483647;border-radius:8px;cursor:pointer;border:2px solid white;text-decoration:none;';
                        document.body.appendChild(btn);
                    }}
                """)
            shot = self.page.screenshot(type="jpeg", quality=70)
            path = self.screenshots_dir / f"{self.model_key}_run{self.run_id}_{prefix}.jpg"
            path.write_bytes(shot)
            return base64.b64encode(shot).decode()
        except Exception:
            return ""

    def extract_page_content(self) -> str:
        try:
            return self.page.locator("body").inner_text()[:4000]
        except Exception:
            return ""

    # ── LLM call ──────────────────────────────────────────────────────────────

    def prompt_llm(
        self,
        pm_data: dict,
        img_b64: str,
        page_text: str,
        current_url: str,
        trajectory: List[dict],
    ) -> tuple:
        """
        Send a single step to the LLM.

        Returns (parsed_action_dict, raw_log_dict).
        The action dict is either a navigation action or a full BDI decision.
        """
        pt           = _page_type(current_url)
        scrolls_done = _count_scrolls_since_last_click(trajectory)
        min_s        = MIN_SCROLLS.get(pt, 0)
        on_sel       = (pt == "selection")
        ready        = on_sel and scrolls_done >= min_s

        page_context = {
            "page_type":   pt,
            "page_text":   page_text,
            "scrolls_done": scrolls_done,
            "min_scrolls": min_s,
            "ready":       ready,
        }

        if self.arch == "react":
            system_prompt = build_react_system_prompt(self.condition, pm_data,
                                                      target_karma=self.target_karma)
            user_text     = build_react_user_message(page_context)
        elif self.arch == "nobdi":
            system_prompt = build_nobdi_system_prompt(self.condition, pm_data,
                                                      target_karma=self.target_karma)
            user_text     = build_nobdi_user_message(page_context)
        else:
            system_prompt = build_system_prompt(self.condition, pm_data,
                                                target_karma=self.target_karma)
            user_text     = build_user_message(
                page_context, harm_assessment=self.harm_assessment
            )

        # Obscure the instance name from the URL so the model reads the page
        obscured_url = "http://localhost:9999/"
        if pt == "dashboard":
            obscured_url += "portfolio_dashboard.html"
        elif pt == "news":
            obscured_url += "news_article.html"
        elif pt == "reddit":
            obscured_url += "reddit_thread/"
        elif pt == "selection":
            obscured_url += "meme_selection.html"

        user_text = f"URL: {obscured_url}\n\n{user_text}"

        # Build message list with proper role separation
        user_content = []
        if img_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        for attempt in range(LLM_RETRIES):
            try:
                request_body = {
                    "model":             self.model,
                    "messages":          messages,
                    "temperature":       TEMPERATURE,
                    "include_reasoning": True,
                }
                if self.run_seed is not None:
                    request_body["seed"] = self.run_seed

                resp = self.client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer":  "http://localhost:3000",
                        "X-Title":       "MemeEconomy-Benchmark",
                    },
                    json=request_body,
                    timeout=LLM_TIMEOUT,
                )

                if resp.status_code != 200:
                    print(f"  [LLM] HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(2)
                    continue

                data      = resp.json()
                msg       = data.get("choices", [{}])[0].get("message", {})
                content   = msg.get("content", "{}") or "{}"
                reasoning = msg.get("reasoning", "") or ""

                raw_log = {
                    "raw_content": content,
                    "reasoning":   reasoning,
                    "model":       self.model,
                    "attempt":     attempt + 1,
                    "usage":       data.get("usage", {}),
                    "system_prompt": system_prompt,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "request_seed": self.run_seed,
                }

                parsed = None

                # ── ReAct: parse Thought+Action text on selection page ────────
                # Accept on any selection step — model often skips scrolling
                if self.arch == "react" and pt == "selection":
                    decision, thought = parse_react_decision(content)
                    if decision:
                        bdi_raw = react_to_bdi_raw(thought, decision)
                        raw_log["react_thought"] = thought
                        raw_log["react_action"]  = decision.get("action_line", "")
                        return bdi_raw, raw_log
                    else:
                        print(f"  [ReAct] Could not parse decision:\n{content[:300]}")

                # ── BDI / navigation: parse JSON ─────────────────────────────
                try:
                    parsed = json.loads(content)
                except Exception:
                    m = re.search(r'\{.*\}', content, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except Exception as e2:
                            print(f"  [LLM] JSON parse error: {e2}")
                    if parsed is None:
                        print(f"  [LLM] Could not parse JSON:\n{content[:300]}")

                if parsed is not None:
                    return parsed, raw_log

            except Exception as e:
                print(f"  [LLM] Request error (attempt {attempt+1}): {e}")
                time.sleep(2)

        raw_log = {
            "raw_content": "",
            "reasoning":   "",
            "model":       self.model,
            "attempt":     LLM_RETRIES,
            "error":       "all retries failed",
            "system_prompt": system_prompt,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "request_seed": self.run_seed,
        }
        return {"action": "error", "reason": "failed to get LLM response"}, raw_log

    # ── Action executor ───────────────────────────────────────────────────────

    def execute_action(self, action: dict, meme_selection_url: str = "",
                       page_type: str = "unknown") -> dict:
        result = {"success": False, "error": None, "new_url": None, "is_terminal": False}

        # On the selection page, accept ANY JSON that contains a meme letter
        # and a subreddit string — the model uses many different key names.
        if page_type == "selection":
            action_str = str(action)
            has_letter = bool(re.search(r'\bOption\s+([A-J])\b', action_str, re.IGNORECASE))
            has_sub    = bool(re.search(r'\br/[a-zA-Z0-9_]+', action_str))
            if has_letter and has_sub:
                result["is_terminal"] = True
                result["decision"]    = action
                result["success"]     = True
                return result

        # Explicit decision wrapper keys on non-selection pages → guard and reject
        is_wrapped_decision = (
            "context_reading"    in action
            or "belief_formation" in action
            or "decision"         in action
        )
        if is_wrapped_decision and page_type != "selection":
            print(f"  [GUARD] Decision rejected on page_type='{page_type}'. Forcing navigation.")
            result["success"] = False
            result["error"]   = f"decision_on_wrong_page:{page_type}"
            return result

        action_type = action.get("action", "").lower()

        try:
            if action_type == "click":
                link_text = action.get("text", "")
                if "Proceed to Meme Selection" in link_text and meme_selection_url:
                    self.page.goto(meme_selection_url)
                    time.sleep(2)
                    result["success"] = True
                    result["new_url"] = self.page.url
                    return result

                # Try link first, then button
                for locator in [
                    self.page.locator(f"a:has-text('{link_text}')").first,
                    self.page.locator(f"button:has-text('{link_text}')").first,
                ]:
                    if locator.count() > 0:
                        locator.click()
                        time.sleep(2)
                        result["success"] = True
                        result["new_url"] = self.page.url
                        return result

                result["error"] = f"Element not found: {link_text}"

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                offset    = 800 if direction == "down" else -800
                self.page.evaluate(f"window.scrollBy(0, {offset})")
                time.sleep(0.8)
                result["success"] = True

            elif action_type == "navigate":
                url = action.get("url", "")
                if not url.startswith("http"):
                    url = BASE_URL + url
                self.page.goto(url)
                time.sleep(2)
                result["success"] = True
                result["new_url"] = self.page.url

            elif action_type == "back":
                self.page.go_back()
                time.sleep(2)
                result["success"] = True
                result["new_url"] = self.page.url

            else:
                result["error"] = f"Unknown action: {action_type}"

        except Exception as e:
            result["error"] = str(e)

        return result


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    model: str,
    num_rounds: int,
    condition: str = "normal",
    headless: bool = True,
    random_instances: bool = False,
    resume_portfolio: Optional[str] = None,
    instance_filter: Optional[List[str]] = None,
    target_karma: Optional[int] = None,
    arch: str = "bdi",
    verifier_key: Optional[str] = None,
    run_label: str = "",
    results_label: str = "",
    run_seed: Optional[int] = None,
    harm_assessment: bool = False,
    split: str = "paper98",
):
    import random as _random

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"MemeEconomy Benchmark")
    print(f"  Model     : {model}  ({MODELS.get(model, model)})")
    print(f"  Condition : {condition}")
    print(f"  Rounds    : {num_rounds}")
    print(f"  Headless  : {headless}")
    print(f"  Run seed  : {run_seed if run_seed is not None else 'legacy/default'}")
    print(f"  Harm eval : {'candidate-level, before intention' if harm_assessment else 'off'}")
    print(f"  Time      : {timestamp}")
    print(f"{'='*60}\n")

    if not OPENROUTER_KEY:
        raise RuntimeError(
            "OPENROUTER_KEY is not set. Refusing to start a paid benchmark run."
        )

    # Load task list
    with open(TASKS_PATH, "r") as f:
        tasks_data = json.load(f).get("injected_tasks", {})

    available = available_instances(tasks_data, split)
    print(f"  Split     : {split} ({len(available)} injected instances)")
    if not available:
        print("[ERROR] No instances with valid Postmill URLs.")
        return

    if instance_filter:
        unknown = [i for i in instance_filter if i not in available]
        if unknown:
            print(f"[WARN] Unknown instances (skipping): {unknown}")
        available = [i for i in instance_filter if i in available]
        if not available:
            print("[ERROR] No valid instances after filtering.")
            return
        num_rounds = len(available)

    if random_instances and not instance_filter:
        instances = _random.sample(available, min(num_rounds, len(available)))
    else:
        instances = [available[i % len(available)] for i in range(num_rounds)]

    # Per-model output directory — optionally namespaced under a results label
    base_out  = RESULTS_DIR / results_label if results_label else RESULTS_DIR
    model_dir = base_out / model
    model_dir.mkdir(parents=True, exist_ok=True)
    run_id   = _next_run_id(model_dir)
    log_path = model_dir / f"run{run_id}_{timestamp}.jsonl"

    def log_step(record: dict):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Resume logic
    if resume_portfolio:
        pm    = PortfolioManager.load_state(resume_portfolio, config_path=str(CONFIG_PATH))
        pm.agent_id     = model
        pm.total_rounds = num_rounds
        skip  = pm.current_round
        instances = instances[skip:]
        print(f"[RESUME] Loaded from {resume_portfolio}, starting at round {skip + 1}")
    else:
        pm   = PortfolioManager(agent_id=model, config_path=str(CONFIG_PATH),
                                total_rounds=num_rounds, condition=condition)
        skip = 0

    pm.pressure_condition = condition

    condition_meta = get_condition_metadata(condition)
    log_step({
        "type":       "run_start",
        "model":      model,
        "model_id":   MODELS.get(model, model),
        "condition":  condition,
        "run_id":     run_id,
        "instances":  instances,
        "num_rounds": num_rounds,
        "timestamp":  timestamp,
        "resumed_from":  resume_portfolio,
        "start_round":   skip + 1,
        "target_karma":  target_karma,
        "run_seed":      run_seed,
        "condition_factors": condition_meta,
        "run_label": run_label,
        "results_label": results_label,
        "harm_assessment": harm_assessment,
    })

    agent = MemeAgent(model, model_dir, condition=condition, run_id=run_id,
                      target_karma=target_karma, arch=arch, run_seed=run_seed,
                      harm_assessment=harm_assessment)
    agent.start_browser(headless=headless)
    agent.login()

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")))

    with open(CONFIG_PATH, "r") as f:
        full_config = json.load(f)
    taxonomy          = full_config.get("subreddit_taxonomy", {})
    all_instances_cfg = full_config.get("instances", {})

    terminal_reason = None
    for idx, instance_name in enumerate(instances, skip + 1):
        print(f"\n{'='*50}")
        print(f"ROUND {idx}/{num_rounds}: {instance_name}")
        print(f"{'='*50}")

        instance_config_data = all_instances_cfg.get(instance_name, {})

        memes, all_subs = _render_round_pages(
            pm, idx, num_rounds, instance_name, model, env,
            tasks_data, taxonomy, instance_config_data,
            run_label=run_label,
        )

        # Build letter → meme_id map for this round
        letter_to_id = {m["letter"]: m["id"] for m in memes}

        # Navigate agent to dashboard
        starting_url = f"{BASE_URL}/test_pages/portfolio_dashboard_{model}{run_label}.html"
        agent.page.goto(starting_url)
        time.sleep(2)

        trajectory: List[dict] = []
        bdi_result = None
        sel_target_url = (
            f"http://localhost:9999/test_pages/"
            f"selection_{model}{run_label}_{instance_name.replace(' ','_').replace('/','_')}_gemini.html"
        )

        # ── Agent loop ────────────────────────────────────────────────────────
        for step in range(MAX_STEPS):
            url_now         = agent.page.url
            screenshot_name = f"round{idx}_step{step+1}"
            print(f"  [Step {step+1:02d}] {url_now}", flush=True)

            img_b64   = agent.screenshot(screenshot_name, meme_selection_url=sel_target_url)
            page_text = agent.extract_page_content()
            pm_data   = pm.get_dashboard_data()

            action_json, raw_log = agent.prompt_llm(
                pm_data, img_b64, page_text, url_now, trajectory,
            )

            # Log reasoning/thinking from reasoning models
            if raw_log.get("reasoning"):
                thinking = raw_log["reasoning"][:400]
                print(f"  [THINKING] {thinking}{'...' if len(raw_log['reasoning'])>400 else ''}")

            # Log scroll compliance without blocking agent
            pt           = _page_type(url_now)
            scrolls_done = _count_scrolls_since_last_click(trajectory)
            min_s        = MIN_SCROLLS.get(pt, 0)
            is_decision  = "context_reading" in action_json or "belief_formation" in action_json or "decision" in action_json
            if is_decision and scrolls_done < min_s:
                print(f"  [SCROLL WARN] decided after only {scrolls_done}/{min_s} scrolls on {pt}")

            print(f"  [ACTION] {json.dumps(action_json, ensure_ascii=False)[:200]}")

            res = agent.execute_action(action_json, meme_selection_url=sel_target_url,
                                        page_type=pt)
            print(f"  [RESULT] success={res.get('success')} | {res.get('error') or res.get('new_url','')}")

            # If the model output a decision on the wrong page, inject a
            # corrective navigation action so next step moves forward.
            if str(res.get("error") or "").startswith("decision_on_wrong_page"):
                nav_actions = {
                    "dashboard": {"action": "click", "text": "GO TO NEWS ARTICLE →"},
                    "news":      {"action": "click", "text": "Go to Reddit Thread"},
                    "reddit":    {"action": "click", "text": "Proceed to Meme Selection →"},
                }
                forced = nav_actions.get(pt)
                if forced:
                    print(f"  [GUARD] Auto-navigating forward from {pt}: {forced}")
                    agent.execute_action(forced, meme_selection_url=sel_target_url,
                                         page_type=pt)

            log_step({
                "type":             "step",
                "round":            idx,
                "instance":         instance_name,
                "condition":        condition,
                "step":             step + 1,
                "url":              url_now,
                "page_type":        pt,
                "scrolls_on_page":  scrolls_done,
                "screenshot":       f"screenshots/{screenshot_name}.jpg",
                "page_text_preview": page_text[:500],
                "raw_llm_content":  raw_log.get("raw_content", ""),
                "reasoning":        raw_log.get("reasoning", ""),
                "model":            raw_log.get("model", ""),
                "usage":            raw_log.get("usage", {}),
                "system_prompt":    raw_log.get("system_prompt", "") if step == 0 else None,
                "system_prompt_sha256": raw_log.get("system_prompt_sha256"),
                "request_seed":     raw_log.get("request_seed"),
                "parsed_action":    action_json,
                "action_result":    res,
            })

            trajectory.append(action_json)

            if res.get("is_terminal"):
                bdi_result = res.get("decision", action_json)
                print(f"  [DECISION at step {step+1}]")
                break

        # ── Extract decision ──────────────────────────────────────────────────
        stated_letter      = None
        resolved_meme_id   = None
        resolved_sub       = None
        resolution_success = False
        risk_acknowledged  = False

        if bdi_result:
            extracted = extract_decision(bdi_result)
            stated_letter     = extracted["stated_letter"]
            resolved_sub      = extracted["target_subreddit"]
            risk_acknowledged = extracted["risk_acknowledged"]

            # Pull any reasoning/context text the model produced
            reasoning_text = extracted.get("reasoning")
            if not reasoning_text:
                # Try common locations in flat or nested BDI
                for key in ("rationale", "reasoning", "justification",
                            "event_summary", "context_reading"):
                    val = bdi_result.get(key)
                    if isinstance(val, str) and len(val) > 20:
                        reasoning_text = val
                        break
                    if isinstance(val, dict):
                        sub = val.get("event_summary") or val.get("reasoning") or ""
                        if len(sub) > 20:
                            reasoning_text = sub
                            break

            if stated_letter and stated_letter in letter_to_id:
                resolved_meme_id   = letter_to_id[stated_letter]
                resolution_success = True
                print(f"  [RESOLVED] Option {stated_letter} → {resolved_meme_id} | sub={resolved_sub}")

                # ── Verifier interception ─────────────────────────────────────
                if verifier_key and bdi_result:
                    safe_letter = next(
                        (l for l, mid in letter_to_id.items()
                         if mid.startswith("funny")), stated_letter
                    )
                    verifier = VerifierAgent(verifier_key)
                    sel_img  = next(
                        (raw_log.get("screenshot_b64") for raw_log in
                         [raw_log] if raw_log.get("screenshot_b64")), None
                    ) if 'raw_log' in dir() else None
                    verdict = verifier.verify(
                        bdi          = bdi_result,
                        img_b64      = sel_img,
                        subreddit    = resolved_sub or "",
                        instance     = instance_name,
                        condition    = condition,
                        karma        = pm.karma_balance,
                        round_num    = idx,
                        meme_options = letter_to_id,
                        safe_letter  = safe_letter,
                        subject_model= model,
                    )
                    label = verdict["safety_label"]
                    print(f"  [VERIFIER] {label} — {verdict['explanation'][:80]}")
                    if label == "BLOCK" and verdict.get("override_letter"):
                        ol = verdict["override_letter"]
                        if ol in letter_to_id:
                            print(f"  [VERIFIER] BLOCK override: {stated_letter}→{ol} ({letter_to_id[ol]})")
                            stated_letter    = ol
                            resolved_meme_id = letter_to_id[ol]
                    bdi_result["_verifier"] = verdict
            else:
                if stated_letter:
                    print(f"  [!] Letter '{stated_letter}' not in meme list {list(letter_to_id.keys())}. Round incomplete.")
                else:
                    print(f"  [!] No meme letter found in BDI. Round incomplete.")
        else:
            print("  [!] No BDI decision produced. Round incomplete.")

        round_incomplete = not resolution_success or resolved_sub is None

        # ── Process round ─────────────────────────────────────────────────────
        round_seed = (
            _stable_round_seed(run_seed, idx, instance_name)
            if run_seed is not None else None
        )
        if not round_incomplete:
            pm.process_round(
                instance_name, resolved_meme_id, resolved_sub,
                bdi_data={
                    **bdi_result,
                    **({"_option_map": dict(letter_to_id)} if harm_assessment else {}),
                    "stated_meme_letter":  stated_letter,
                    "resolution_success":  resolution_success,
                    "risk_acknowledged":   risk_acknowledged,
                    "reasoning":           reasoning_text,
                },
                seed=round_seed,
            )

        # ── Log round result ──────────────────────────────────────────────────
        dash = pm.get_dashboard_data()
        last = dash["round_history"][-1] if (not round_incomplete and dash["round_history"]) else {}

        log_step({
            "type":              "round_result",
            "round":             idx,
            "instance":          instance_name,
            "condition":         condition,
            "meme_chosen":       resolved_meme_id,
            "stated_letter":     stated_letter,
            "resolution_success": resolution_success,
            "subreddit":         resolved_sub,
            "risk_acknowledged": risk_acknowledged,
            "karma_delta":       last.get("karma_delta", 0),
            "karma_balance":     dash["karma"]["balance"],
            "meme_tier":         last.get("meme_tier"),
            "subreddit_fit":     last.get("subreddit_fit"),
            "removal":           last.get("removal_triggered", False),
            "steps_taken":       len(trajectory),
            "incomplete":        round_incomplete,
            "bdi_result":        bdi_result,
            "option_map":        dict(letter_to_id) if harm_assessment else None,
            "random_seed":       round_seed,
        })

        delta   = last.get("karma_delta", 0)
        balance = dash["karma"]["balance"]
        status  = "INCOMPLETE" if round_incomplete else ("REMOVED" if last.get("removal_triggered") else f"{delta:+,}")
        print(f"  [ROUND END] {status}  balance={balance:,}")

        threshold = condition_meta.get("elimination_threshold")
        if threshold is not None and balance <= threshold:
            terminal_reason = "elimination_threshold"
            log_step({
                "type": "terminal_event",
                "condition": condition,
                "round": idx,
                "instance": instance_name,
                "reason": terminal_reason,
                "threshold": threshold,
                "karma_balance": balance,
            })
            print(f"  [TERMINAL] Karma reached {balance:,} <= {threshold:,}")
            break

    # ── Save final state ──────────────────────────────────────────────────────
    portfolio_path = model_dir / f"portfolio_run{run_id}_{timestamp}.json"
    pm.save_state(str(portfolio_path))
    log_step({
        "type":          "run_end",
        "condition":     condition,
        "portfolio_file": str(portfolio_path),
        "final_karma":   pm.karma_balance,
        "completed_rounds": pm.current_round,
        "planned_rounds": num_rounds,
        "terminal_reason": terminal_reason,
        "run_seed": run_seed,
        "timestamp":     datetime.now().isoformat(),
    })
    agent.stop_browser()

    print(f"\nBenchmark complete!")
    print(f"  Portfolio : {portfolio_path}")
    print(f"  Step log  : {log_path}")


# ── Human mode ────────────────────────────────────────────────────────────────

def run_human_benchmark(
    num_rounds: int = 10,
    condition: str = "normal",
    random_instances: bool = False,
    split: str = "paper98",
):
    import random as _random

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"MemeEconomy — HUMAN MODE  (condition: {condition})")
    print(f"{'='*60}\n")
    print("Workflow each round:")
    print("  Browser opens at Dashboard → click to News → Reddit → Meme Selection")
    print("  Come back here and type your pick when done.\n")

    with open(TASKS_PATH, "r") as f:
        tasks_data = json.load(f).get("injected_tasks", {})

    available = available_instances(tasks_data, split)
    if not available:
        print("[ERROR] No instances with valid Postmill URLs.")
        return

    if random_instances:
        instances = _random.sample(available, min(num_rounds, len(available)))
    else:
        instances = [available[i % len(available)] for i in range(num_rounds)]

    log_dir  = RESULTS_DIR / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"human_{timestamp}.jsonl"

    def log_step(record: dict):
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    log_step({
        "type": "run_start", "model": "human", "condition": condition,
        "instances": instances, "num_rounds": num_rounds, "timestamp": timestamp,
    })

    pm = PortfolioManager(agent_id="human", config_path=str(CONFIG_PATH),
                          total_rounds=num_rounds, condition=condition)
    pm.pressure_condition = condition

    from playwright.sync_api import sync_playwright
    pw      = sync_playwright().start()
    browser = pw.chromium.launch(headless=False)
    context = browser.new_context(viewport={"width": 1400, "height": 900})
    page    = context.new_page()

    page.goto(f"{BASE_URL}/login", timeout=30_000)
    try:
        if page.locator("input[name='_username']").count() > 0:
            page.fill("input[name='_username']", "MarvelsGrantMan136")
            page.fill("input[name='_password']", "test1234")
            page.click("button[type='submit']")
            page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:
        pass

    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "templates")))

    with open(CONFIG_PATH, "r") as f:
        full_config = json.load(f)
    taxonomy          = full_config.get("subreddit_taxonomy", {})
    all_instances_cfg = full_config.get("instances", {})

    for idx, instance_name in enumerate(instances, 1):
        print(f"\n{'='*50}")
        print(f"ROUND {idx}/{num_rounds}: {instance_name}")
        print(f"{'='*50}")

        instance_config_data = all_instances_cfg.get(instance_name, {})
        memes, all_subs = _render_round_pages(
            pm, idx, num_rounds, instance_name, "human", env,
            tasks_data, taxonomy, instance_config_data,
        )

        letter_to_id = {m["letter"]: m["id"] for m in memes}
        sub_names    = [s["name"] for s in all_subs]

        page.goto(f"{BASE_URL}/test_pages/portfolio_dashboard_human.html")
        tier_names = {1: "Funny", 2: "Poor", 3: "Neutral", 4: "Noise", 5: "EDGY"}
        print("\nMemes:")
        for m in memes:
            print(f"  Option {m['letter']}  [{tier_names.get(m['tier'], '?')}]  id={m['id']}")
        print("\nSubreddits:")
        print("  " + "  ".join(sub_names))
        print("\nPress ENTER when you've browsed and are ready to pick.")
        input()

        valid_letters = list(letter_to_id.keys())
        while True:
            raw = input(f"Meme choice ({'/'.join(valid_letters)}): ").strip().upper()
            if raw in valid_letters:
                break
            print(f"  Pick one of: {valid_letters}")

        while True:
            raw_sub = input("Subreddit (e.g. r/memes): ").strip()
            if not raw_sub.startswith("r/"):
                raw_sub = "r/" + raw_sub
            if raw_sub in sub_names:
                break
            confirm = input(f"'{raw_sub}' not in displayed list. Use anyway? (y/n): ").strip().lower()
            if confirm == "y":
                break

        chosen_meme_id = letter_to_id[raw]
        human_bdi      = {
            "human_choice":       True,
            "selected_meme":      f"Option {raw}",
            "target_subreddit":   raw_sub,
            "stated_meme_letter": raw,
            "resolution_success": True,
            "risk_acknowledged":  False,
        }

        pm.process_round(instance_name, chosen_meme_id, raw_sub, bdi_data=human_bdi)

        dash = pm.get_dashboard_data()
        last = dash["round_history"][-1] if dash["round_history"] else {}

        log_step({
            "type":              "round_result",
            "round":             idx,
            "instance":          instance_name,
            "condition":         condition,
            "meme_chosen":       chosen_meme_id,
            "stated_letter":     raw,
            "resolution_success": True,
            "subreddit":         raw_sub,
            "risk_acknowledged": False,
            "karma_delta":       last.get("karma_delta", 0),
            "karma_balance":     dash["karma"]["balance"],
            "meme_tier":         last.get("meme_tier"),
            "subreddit_fit":     last.get("subreddit_fit"),
            "removal":           last.get("removal_triggered", False),
            "incomplete":        False,
            "bdi_result":        human_bdi,
        })

        print(f"  karma_delta={last.get('karma_delta',0):+,}  balance={dash['karma']['balance']:,}")

    portfolio_path = RESULTS_DIR / f"portfolio_human_{timestamp}.json"
    pm.save_state(str(portfolio_path))
    log_step({
        "type": "run_end", "condition": condition,
        "portfolio_file": str(portfolio_path),
        "final_karma": pm.karma_balance,
        "timestamp": datetime.now().isoformat(),
    })
    browser.close()
    pw.stop()
    print(f"\nHuman session complete!")
    print(f"  Portfolio : {portfolio_path}")
    print(f"  Step log  : {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemeEconomy Benchmark Runner")

    parser.add_argument("--models",     nargs="+", default=["qwen32b"],
                        help="Model key(s) from MODELS dict")
    parser.add_argument("--rounds",     type=int,  default=10,
                        help="Number of instances per model")
    parser.add_argument("--condition",  choices=list(CONDITIONS), default="normal",
                        help="Pressure condition framing for the system prompt")
    parser.add_argument("--visible",    action="store_true",
                        help="Show browser window")
    parser.add_argument("--random",     action="store_true",
                        help="Pick instances randomly instead of sequentially")
    parser.add_argument("--human",      action="store_true",
                        help="Human mode: browse manually and type picks")
    parser.add_argument("--list-models", action="store_true",
                        help="Print all model keys and exit")
    parser.add_argument("--resume-portfolio", type=str, default=None, metavar="PATH",
                        help="Path to existing portfolio JSON to resume from")
    parser.add_argument("--instances",  nargs="+", default=None, metavar="INSTANCE",
                        help="Run only these specific instance names")
    parser.add_argument("--target-karma", type=int, default=None, metavar="KARMA",
                        help="Custom karma target (only used with --condition dose_response)")
    parser.add_argument("--arch", choices=["bdi", "react", "nobdi"], default="bdi",
                        help="Agent architecture: bdi (default), react, or nobdi")
    parser.add_argument("--verifier", type=str, default=None, metavar="MODEL_KEY",
                        help="Model key to use as real-time verifier (e.g. gpt5_1)")
    parser.add_argument("--run-label", type=str, default="", metavar="LABEL",
                        help="Suffix appended to HTML page filenames (e.g. _abl2) — "
                             "lets two runs of the same model coexist without clobbering each other's pages")
    parser.add_argument("--results-label", type=str, default="", metavar="LABEL",
                        help="Subfolder under results/ for all output files (e.g. abl4) — "
                             "results land in results/LABEL/{model}/ instead of results/{model}/")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Path to alternate instance_config.json (e.g. configs/instance_config_20pct.json)")
    parser.add_argument("--split", choices=SPLITS, default="paper98",
                        help="Instance split: paper98 (default; the 98 used in the paper), "
                             "extension (added after the paper), or all")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Stable run seed; matched conditions share this seed")
    parser.add_argument("--harm-assessment", action="store_true",
                        help="Require a candidate-level harm classification in the BDI belief block before intention")

    args = parser.parse_args()

    # Override CONFIG_PATH if --config is provided.
    # NOTE: do NOT do `import memeeconomy.run_single_agent as _self` here. When this file
    # is executed as a script it is the module `__main__`; importing it by name loads a
    # SECOND copy, so assigning to that copy's CONFIG_PATH leaves the running module
    # untouched and the override silently does nothing. Assign to this module's own
    # globals instead, which is what run_benchmark() reads at call time.
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"[CONFIG] file not found: {cfg_path}")
        globals()["CONFIG_PATH"] = cfg_path
        import json as _json
        _rr = _json.load(open(cfg_path)).get("removal_risk", {})
        print(f"[CONFIG] Using alternate config: {cfg_path} "
              f"(tier5_removal_chance={_rr.get('tier5_removal_chance')})")

    if args.list_models:
        print("Available models:")
        for k, v in MODELS.items():
            print(f"  {k:20s} → {v}")
        raise SystemExit(0)

    if args.human:
        run_human_benchmark(
            num_rounds=args.rounds,
            condition=args.condition,
            random_instances=args.random,
            split=args.split,
        )
    else:
        for m in args.models:
            if m not in MODELS:
                print(f"[WARN] Unknown model key '{m}'. Will pass string directly to OpenRouter.")
            run_benchmark(
                model=m,
                num_rounds=args.rounds,
                condition=args.condition,
                headless=not args.visible,
                random_instances=args.random,
                resume_portfolio=args.resume_portfolio,
                instance_filter=args.instances,
                target_karma=args.target_karma,
                arch=args.arch,
                verifier_key=args.verifier,
                run_label=args.run_label,
                results_label=args.results_label,
                run_seed=args.seed,
                harm_assessment=args.harm_assessment,
                split=args.split,
            )
