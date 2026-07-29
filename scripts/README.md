# scripts

Analysis and evaluation scripts for the scrapegraph-100k paper. All of them pin the datasets to revision `v1.0` (`scrapegraphai/scrapegraphai-100k` raw, `scrapegraphai/scrapegraph-100k-finetuning` splits) and write tables into `tables/`.

Run everything from the repo root with `uv run python scripts/<name>.py`. Scripts that call model APIs read `API_BASE` / `API_KEY` from `.env` (LiteLLM proxy or any OpenAI-compatible endpoint), without them they use the provider's default credentials.

## graphs.py

Publication figures and descriptive-stats tables for the raw dataset: schema complexity distributions, response sizes, model distribution, validation vs complexity, correlation matrix, model scaling. No CLI flags.

```sh
uv run python scripts/graphs.py
```

Outputs: PNGs in `figures/`, CSVs in `tables/`.

## analysis.py

Language identification over dataset content with fastText (`facebook/fasttext-language-identification`), plus the language-distribution figure. Shares style/config with `graphs.py`. No CLI flags.

```sh
uv run python scripts/analysis.py
```

Outputs: `tables/language_stats.csv`, `figures/content_language_dist.png`.

## frontier_baselines.py

Frontier structured-output baselines via LiteLLM (reviewer Q3, deliverable D4). Evaluates API models with native structured-output mode on the same test subset used for the fine-tuned models (the 8192-token prompt filter from
`modelling/evaluation.py`) and reports metrics per schema-complexity bucket — the same bins as `graphs.py:validation_vs_complexity` — with explicit rows for the depth >= 7 and keys >= 200 tails. "Schema rejected by API" is tracked as its own outcome, distinct from generating invalid output: on rejection the call falls back to plain `json_object` mode so every sample still gets a prediction. Complexity score, depth and key counts come from the raw dataset, joined to test rows by exact schema string.

```sh
uv run python scripts/frontier_baselines.py --model gpt-5-mini
uv run python scripts/frontier_baselines.py --model gemini/gemini-2.5-flash --sample 500
```

Flags: `--structured-mode json_schema|json_object|none`, `--sample N` (stratified, 0 = full set), `--concurrency`, `--seed`, `--max-new-tokens`, `--temperature` (omitted from requests by default), `--extra-body '<json>'` (forwarded verbatim, e.g. vLLM sampling params), `--no-length-filter`, `--resume` (continue an interrupted run from its JSONL), `--gold <gold.jsonl>` (evaluate only the human-benchmark rows; results get a `_gold` suffix so full-test-set runs are never clobbered — feed the JSONL to `human_benchmark.py score --results` for semantic correctness against human gold).

Outputs: `sg-checkpoints/results/litellm_<model>.jsonl` (per-sample records), `litellm_<model>_summary.json`, `tables/frontier_<model>_by_complexity.md`.

## regeneration_comparison.py

Compares the raw dataset's original responses against the regenerated responses in the fine-tuning repo, over byte-identical content chunks (chunking mirrors `modelling/preprocess.py` exactly). Classifies per-leaf changes and computes BLEU-based soft metrics on a 5,000-pair sample. No CLI flags.

```sh
uv run python scripts/regeneration_comparison.py
```

## rejection_report.py

Post-hoc schema-rejection accounting for a `frontier_baselines.py` run (reviewer follow-up on the Claude results). Reads the per-sample JSONL — no re-inference — and reports, per complexity bucket plus the depth/keys tails: total vs API-accepted counts, metrics over all completed calls (the paper's denominator: rejected schemas fall back to `json_object`, so every call is scored), accepted-only metrics, and a strict end-to-end metric that treats provider-side schema rejection as failure. Also prints the normalized top rejection causes parsed from the logged error strings and the run configuration from the sibling `_summary.json`. `--reference` takes another run's JSONL covering the full planned index set (e.g. the gpt run) to quantify coverage, an end-to-end metric that also counts never-completed calls as failure, and a coverage-sensitivity check (the reference model's metrics on its full set vs restricted to the indices this run completed — small deltas mean the covered subset is representative, so an incomplete run doesn't need resuming).

```sh
uv run python scripts/rejection_report.py --results sg-checkpoints/results/litellm_claude-sonnet-4.6-fiit.jsonl --reference sg-checkpoints/results/litellm_gpt-5.6-terra-fiit.jsonl
```

Flags: `--results` (required), `--reference`, `--top-causes N` (default 10).

Outputs: `tables/rejection_report_<model>.md`.

## rewrite_taxonomy.py

Splits `regeneration_comparison.py`'s `chg_content_rewrite` class (reported in the rebuttal as "genuine re-extractions or paraphrases") into meaning-preserving rewrites vs substantive value changes. Three stages: `collect` rebuilds the aligned pairs, extracts every changed shared leaf classified as a content rewrite, and settles the deterministic cases (punctuation/diacritics/number/date formatting, list reorder, strict containment = truncation/completion); `judge` sends a random sample of the residual leaves to an LLM judge that labels each pair equivalent / partial / substantive / unclear (judgments are cached and resumable, error records retried); `analyze` combines heuristic counts with the extrapolated judged split into the final table.

```sh
uv run python scripts/rewrite_taxonomy.py collect
uv run python scripts/rewrite_taxonomy.py judge --n-judge 1500
uv run python scripts/rewrite_taxonomy.py analyze
```

The `sheet` stage is the zero-API alternative to `judge`: it writes the same seeded residual sample to `manual_sheet.csv` for hand-labeling (fill the `verdict` column with equivalent/partial/substantive/unclear); `analyze` merges labeled rows with any judge output, manual labels winning on conflict.

```sh
uv run python scripts/rewrite_taxonomy.py sheet --n-judge 200
uv run python scripts/rewrite_taxonomy.py analyze
```

Stages: `collect`, `sheet`, `judge`, `analyze`, `all` (default = collect+judge+analyze). Flags: `--n-judge` (default 1500; also sizes the sheet), `--judge-model` (default `gpt-5.6-terra-fiit`), `--workers`, `--limit`, `--seed`.

Outputs: working files in `.data/rewrite_taxonomy/` (leaves, judgments, manual sheet), `tables/rewrite_taxonomy.md`.

## llm_judge_pilot.py

LLM-judge pilot over regeneration pairs: samples judge units, runs a judge model, and analyzes agreement (including a judge-stability subsample). Stages run individually or all at once.

```sh
uv run python scripts/llm_judge_pilot.py
```

Stages: `sample`, `judge`, `analyze`, `all` (default). Flags: `--n-main`, `--n-pairs`, `--judge-model` (default `gpt-5.6-terra-fiit`), `--second-judge`, `--stability-n`, `--workers`, `--limit`, `--seed`. 

Outputs: working files in `.data/llm_judge_pilot/` (units, judgments), `tables/llm_judge_pilot.csv`.

## human_benchmark.py

Human gold-label benchmark on the test split. Three subcommands:

```sh
uv run python scripts/human_benchmark.py sample --n 100      # stratified leak-free sample
uv run python scripts/human_benchmark.py build               # collect verified annotations into gold.jsonl
uv run python scripts/human_benchmark.py score --results <per-sample results.jsonl> --name <model>
uv run python scripts/human_benchmark.py score --teacher     # score the GPT-5-nano draft targets
```

`sample` writes annotation drafts plus `INSTRUCTIONS.md` and `manifest.json` to `.data/human_benchmark/`, `build` produces `gold.jsonl` and `tables/human_benchmark_build.md`, `score` writes `tables/human_benchmark_<name>.md`. Reuses the bucketing/stratification from `frontier_baselines.py` and the chunking from `regeneration_comparison.py`.

## domain_diversity.py

Domain-diversity statistics over the raw dataset's source URLs: unique root domains, schema/domain concentration, and the top-domains table. No CLI flags.

```sh
uv run python scripts/domain_diversity.py
```

Outputs: `tables/domain_diversity_summary.csv`, `tables/top_domains.csv`.

## pii_audit.py

PII audit of all four dataset fields (`prompt`, `content`, `schema`, `response`) using the OPF PII model plus high-precision regexes for credential formats (AWS/OpenAI/Stripe/GitHub keys, JWTs, private-key blocks, ...). Checkpoints progress so long runs can resume.

```sh
uv run python scripts/pii_audit.py --device cuda
uv run python scripts/pii_audit.py --resume
```

Flags: `--out-dir` (default `pii_audit_out/`), `--device cuda|cpu`, `--limit N` (scan only the first N rows), `--resume`.

Outputs: per-label span CSVs and `audit_table.md` in the out dir.
