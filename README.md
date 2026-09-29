# MemeEconomy project page

Project page for *MemeEconomy: Do LLM Agents Trade Ethics for Survival?* (NeurIPS 2026).
Plain static HTML/CSS, served by GitHub Pages. No build step is needed to serve it.

## Layout
- `index.html` – the page. Blocks between `<!-- @build:NAME -->` markers are generated; edit everything else by hand.
- `static/css/style.css`, `static/js/main.js` – styles and the Tier-5 click-to-reveal.
- `static/img/figures/` – paper figures (WebP), `static/img/memes/` – taxonomy and demo memes (WebP).
- `scripts/build.py` – regenerates the marked blocks and images.

## Regenerating
```
python3 scripts/build.py
```
It parses the tables straight from `neurips_2026.tex` (single-agent, tournament, verifier,
pressure decomposition), copies demo BDI traces verbatim from `results/<model>/portfolio_run*.json`,
and converts figures and memes to WebP. Source paths are set at the top of the script.

## Preview
```
python3 -m http.server 8000   # then open http://localhost:8000
```
