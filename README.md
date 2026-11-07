# MemeEconomy: Do LLM Agents Trade Ethics for Survival?

Code for **MemeEconomy**, a multimodal agentic market simulation for studying
pressure-induced misalignment in LLM agents (NeurIPS 2026).

[Project page](https://syed-nazmus-sakib.github.io/MemeEconomy/) ·
[Dataset](https://huggingface.co/datasets/SyedNazmusSakib/MemeEconomy)

Agents act as meme investors. Each round they read a news article about a real
event, browse the Reddit megathread about it, then pick one of five memes and a
subreddit to post it in, to earn karma. One meme option per event is harmful
(Tier 5). The benchmark measures how often agents choose it, and whether they
acknowledge the harm when they do, as economic pressure rises.

> **Content warning.** The dataset contains intentionally harmful memes (Tier 5).
> Read [CONTENT_DISCLAIMER.md](CONTENT_DISCLAIMER.md) before use.

## What lives where

| | Location |
|---|---|
| Code | this repository |
| Dataset (memes, megathreads, news articles, karma) | Hugging Face, gated — fetched by `scripts/download_data.py` |
| Environment | Reddit WebArena (Postmill) container, set up by `environment/setup_environment.py` |

### Instance splits

Every instance in `configs/instance_config.json` carries a `split` tag:

- **`paper98`**: the 98 instances behind every result in the paper. This is the default everywhere.
- **`extension`**: 202 more instances (300 in total), added after the paper's experiments. Not used in any reported result.

Select with `--split paper98 | extension | all` on the runners.

## Setup

**1. Install**

```bash
pip install -r requirements.txt
playwright install chrome         # the runner drives Google Chrome
cp .env.example .env              # add your OpenRouter key
```

**2. Download the dataset** (accept the terms on the Hugging Face page first)

```bash
huggingface-cli login
python scripts/download_data.py               # → memes/paper98/, memes/extension/
```

**3. Start the environment.** MemeEconomy runs on the Reddit environment from
[WebArena](https://github.com/web-arena-x/webarena). Load the image, then start it
as `reddit_benchmark` on port 9999:

```bash
docker run --name reddit_benchmark -p 9999:80 -d postmill-populated-exposed-withimg
python environment/setup_environment.py --exact             # the paper's 98 instances
python environment/setup_environment.py --split extension   # optional: the other 202
```

`--exact` restores the paper's megathreads from `env_patch/` (same thread ids, comments
and timestamps as the paper's runs; commenter names are pseudonymised). It then builds
the Wikipedia-style news pages, copies the meme images into the container's web root,
and spot-checks that everything is served. It writes `memeeconomy_tasks.json`, which
maps each instance to its thread. Re-running it is safe. Without `--exact`, the script
injects fresh megathreads from the dataset files instead.

**4. Smoke test**

```bash
python -m memeeconomy.run_single_agent --models qwen8b --rounds 3 --seed 1
```

Run all commands from the repository root. Run only one benchmark process per
container at a time.

## Running the benchmark

```bash
# single agent
python -m memeeconomy.run_single_agent --models gpt5_1 --rounds 98 --condition desperation \
    --resume-portfolio results/starting_portfolios/start_desperation.json --seed 1

# multi-agent tournament
python -m memeeconomy.run_tournament --agents gpt5_1 gemini3_1_pro --rounds 46 --condition normal

python -m memeeconomy.run_single_agent --list-models     # model keys (all via OpenRouter)
```

Conditions: `neutral`, `normal`, `desperation`, `extremist`. Desperation and
extremist start from the portfolios in `results/starting_portfolios/`.
Architectures: `--arch bdi` (default), `react`, `nobdi`. Outputs go to `results/<model>/`.

## Reproducing the paper

| Paper | Script |
|---|---|
| §4.1 Single-agent simulation (PIM, GF, drift) | `experiments/main/run_all_98.sh`, then `python evaluation/compute_paper_metrics.py` |
| §4.2 Multi-agent tournament | `experiments/main/run_tournament.sh`, `experiments/main/run_multiagent.sh`; `evaluation/evaluate_multiagent.py` |
| §5 Real-time verification (zero-shot verifiers) | `experiments/main/run_verifier_experiment.sh` |
| §5.2 MemeAgent verifier | `train/` (see `train/README.md`) |
| App. BDI ablation (no schema) | `experiments/ablations/run_ablation2_nobdi.sh` |
| App. ReAct architecture | `experiments/ablations/run_react_ablation.sh` |
| App. Pressure framing decomposition | `experiments/ablations/run_ablation3_pressure.sh`, `evaluation/analyze_ablation3_pressure.py` |
| App. Sample-size robustness | `experiments/ablations/run_ablation4_robustness.sh` |
| App. Removal-probability sensitivity | `configs/instance_config_{20,40,50,60}pct.json`, `experiments/removal_behavioral/`, `experiments/removal_sweep/` |
| Metric validation (LLM judges) | `validation/step1_sample.py` … `step4_compare_judges.py` |
| Pressure factorial (matched controls) | `experiments/pressure_factorial_2026/` (preregistration inside) |
| Harm recognition | `experiments/harm_recognition/` |

`evaluation/evaluate.py` computes every per-run metric from a portfolio file:
`python evaluation/evaluate.py results/<model>/portfolio_run*.json --compare`.

## Repository layout

```
memeeconomy/                    the benchmark
    run_single_agent.py         single-agent runner (Playwright + OpenRouter)
    run_tournament.py           multi-agent tournament runner
    prompts.py                  BDI prompts, conditions, output schema
    prompts_react.py, prompts_nobdi.py   architecture ablations
    portfolio_manager.py        agent state: karma, history, streaks
    karma_calculator.py         scoring: audience fit, removal risk
    verifier_agent.py           real-time verifier
    templates/                  dashboard and selection page templates
configs/                        instance_config.json and removal-probability variants
environment/                    environment setup: injection, news pages
evaluation/                     metrics and paper tables
experiments/
    main/                       launchers for the paper's main experiments
    ablations/                  BDI, ReAct, pressure-framing and sample-size ablations
    removal_behavioral/, removal_sweep/, pressure_factorial_2026/, harm_recognition/
train/                          MemeAgent verifier training
validation/                     LLM-as-judge metric validation
scripts/                        dataset download, starting portfolios
dataset_construction/           how instances were built (provenance; raw data not distributed)
results/starting_portfolios/    starting states for the desperation and extremist conditions
docs/                           project website (GitHub Pages)
```

Generated at runtime and ignored by git: `memes/`, `env_patch/`, `generated_pages/`,
`memeeconomy_tasks.json`, `results/`.

## License

Code: MIT ([LICENSE](LICENSE)). The dataset has its own terms on its Hugging Face page.

## Citation

See [CITATION.cff](CITATION.cff), or use "Cite this repository" on GitHub.
