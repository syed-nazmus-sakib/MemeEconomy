# Content Disclaimer

## Purpose

This repository contains research code and a dataset developed for the purpose of studying AI safety and alignment — specifically, whether large language models exhibit unsafe or harmful behavior when operating under simulated economic pressure in a controlled benchmark environment.

## Nature of the Dataset

The MemeEconomy benchmark dataset includes:

1. **Real-world event metadata.** Descriptions of real news and internet-culture events drawn from publicly available Reddit discussions and news articles (the 98 used in the paper are tagged `split: paper98` in `instance_config.json`). These events include politically sensitive topics, allegations of criminal conduct, and international conflicts.

2. **Meme images.** Images collected from Reddit, including images in five content tiers. **Tier 5 ("edgy") images** contain content that is intentionally offensive, dehumanising, or harmful — including imagery targeting real individuals, mocking real tragedies, referencing violence, or containing material that would be removed under standard platform policies.

3. **Simulated Reddit discussions.** Synthetic megathread content generated to support agent navigation through the benchmark.

## Why Harmful Content Is Included

The central research question of this benchmark is whether AI agents select harmful content when placed under financial pressure. To measure this behavior, the dataset must contain genuinely harmful meme options for agents to either select or avoid. The presence of Tier 5 content is **essential to the benchmark design** — without it, there is no harm to measure.

**Tier 5 images are not included to promote, celebrate, or normalise the content they depict.** They are included solely to provide agents with a choice that carries measurable harm, enabling researchers to study whether economic incentives cause models to override their safety training.

## Responsible Use

This codebase and dataset are released for **academic research purposes only**. Users are expected to:

- Use this benchmark only in controlled research environments.
- Not deploy the benchmark as a live product or consumer-facing application.
- Not extract Tier 5 content from the dataset for redistribution or use outside of the benchmark context.
- Comply with the terms of service of any AI API used to run agents (OpenRouter, OpenAI, Google, etc.).
- Follow their institution's IRB/ethics review requirements when using this benchmark.

## Sensitive Topics

Several benchmark instances involve politically sensitive or legally sensitive real-world events, including but not limited to:

- Active armed conflicts (Palestine/Israel, Iran/USA)
- Allegations of criminal conduct against named public figures
- Politically divisive figures and events in multiple countries
- Content referencing sexual conduct, violence, or personal tragedy

These topics are included because they are precisely the events where harmful meme selection is most consequential. The benchmark is designed to probe model behavior at the ethical boundary, not to take positions on the events themselves.

## No Warranty

The authors make no representations about the suitability of this benchmark for any specific research question. Results produced by this benchmark reflect behavior in a specific simulated economic context and should not be generalised without appropriate caveats.

---

*If you have questions about responsible use, please open an issue on the repository.*
