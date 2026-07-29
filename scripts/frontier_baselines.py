import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import litellm
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelling import metrics as M  # noqa: E402
from modelling import prompts  # noqa: E402
from modelling.utils import extract_json  # noqa: E402

load_dotenv()

# gpt-5 family rejects temperature=0; let litellm drop unsupported params
litellm.drop_params = True

RESULTS_DIR = Path("sg-checkpoints/results")
TABLES_DIR = Path("tables")
RAW_REPO = "scrapegraphai/scrapegraphai-100k"
FT_REPO = "scrapegraphai/scrapegraph-100k-finetuning"
REVISION = "v1.0"

# Same bins as graphs.py validation_vs_complexity
COMPLEXITY_BINS = [0, 50, 100, 200, 500, 1000, float("inf")]
BUCKET_LABELS = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
UNKNOWN_BUCKET = "unknown"

# Production structured-output APIs degrade/reject past these (reviewer Q3)
DEPTH_THRESHOLD = 7
KEYS_THRESHOLD = 200
DEPTH_GROUP = f"depth>={DEPTH_THRESHOLD}"
KEYS_GROUP = f"keys>={KEYS_THRESHOLD}"

# Same limit as modelling/evaluation.py (vLLM eval kept prompts <= 8192 tokens)
MAX_PROMPT_TOKENS = 8192
FILTER_TOKENIZER = "Qwen/Qwen3-4B"

# metrics.run_all always returns exactly these keys
METRIC_KEYS = ["is_valid_json", "is_schema_compliant", *M.magic_metric_failed()]


@dataclass
class FrontierEvalConfig:
    model_name: str = "gpt-5-mini"
    structured_mode: str = "json_schema"  # json_schema | json_object | none
    # Custom OpenAI-compatible / LiteLLM proxy endpoint (API_BASE / API_KEY in .env)
    api_base: str | None = None
    api_key: str | None = None
    # Higher than the vLLM eval's 4096: API baselines shouldn't fail via
    # truncation (dataset responses are <=10k chars, ~3k tokens)
    max_new_tokens: int = 8192
    # None = don't send the param (gpt-5.x models only accept their default)
    temperature: float | None = None
    # Extra request-body fields forwarded verbatim (e.g. vLLM sampling params)
    extra_body: dict | None = None
    timeout: int = 300
    concurrency: int = 16
    sample: int = 0  # 0 = full test set, N = stratified sample of N
    seed: int = 42
    length_filter: bool = True
    resume: bool = False
    # path to a human_benchmark gold.jsonl: restrict inference to its test_idx
    # rows (they already passed the length filter when sampled)
    gold: str | None = None


def bucket_for_score(score: float | None) -> str:
    if score is None:
        return UNKNOWN_BUCKET
    for lo, hi, label in zip(
        COMPLEXITY_BINS[:-1], COMPLEXITY_BINS[1:], BUCKET_LABELS, strict=True
    ):
        if lo <= score < hi:
            return label
    return BUCKET_LABELS[-1]


def load_schema_stats() -> dict[str, dict]:
    """schema string -> {complexity, depth, keys} from the raw dataset."""
    raw = load_dataset(RAW_REPO, split="train", revision=REVISION)
    assert isinstance(raw, Dataset)
    raw = raw.select_columns(
        ["schema", "schema_complexity_score", "schema_depth", "schema_keys"]
    )
    return {
        s: {"complexity": c, "depth": d, "keys": k}
        for s, c, d, k in zip(
            raw["schema"],
            raw["schema_complexity_score"],
            raw["schema_depth"],
            raw["schema_keys"],
            strict=True,
        )
    }


def length_filter_indices(ds) -> list[int]:
    """Reproduce modelling/evaluation.py's prompt-length filter exactly."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FILTER_TOKENIZER)
    kept = []
    for i, row in enumerate(ds):
        prompt = prompts.build(row["schema"], row["content"])
        msgs = [{"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(tokens) <= MAX_PROMPT_TOKENS:
            kept.append(i)
    print(f"Length filter kept {len(kept)}/{len(ds)} samples")
    return kept


def stratified_sample(
    indices: list[int], buckets: dict[int, str], n: int, seed: int
) -> list[int]:
    """Proportional stratified sample over complexity buckets (>=1 per bucket)."""
    if n <= 0 or n >= len(indices):
        return indices
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_bucket[buckets[idx]].append(idx)

    rng = random.Random(seed)
    total = len(indices)
    quotas = {b: max(1, round(n * len(idxs) / total)) for b, idxs in by_bucket.items()}
    # Fix rounding drift so the sample sums to exactly n: trim overshoot from
    # the largest quotas, fill undershoot from buckets with spare capacity
    while sum(quotas.values()) > n:
        largest = max(quotas, key=lambda b: quotas[b])
        quotas[largest] -= 1
    while sum(quotas.values()) < n:
        spare = {b: len(idxs) - quotas[b] for b, idxs in by_bucket.items()}
        grow = max(spare, key=lambda b: spare[b])
        if spare[grow] <= 0:
            break
        quotas[grow] += 1

    sampled = []
    for b, idxs in by_bucket.items():
        sampled.extend(rng.sample(idxs, min(quotas[b], len(idxs))))
    return sorted(sampled)


def response_format_kwargs(mode: str, schema_obj: dict) -> dict:
    if mode == "json_schema":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": schema_obj,
                    "strict": False,
                },
            }
        }
    if mode == "json_object":
        return {"response_format": {"type": "json_object"}}
    return {}


async def call_model(
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    schema_obj: dict,
    config: FrontierEvalConfig,
) -> dict:
    """Returns {content, structured_mode_used, schema_rejected, cost, error}."""
    mode = config.structured_mode
    schema_rejected = False
    rejection_error = ""
    model = config.model_name
    call_kwargs = {
        "max_tokens": config.max_new_tokens,
        "timeout": config.timeout,
    }
    if config.temperature is not None:
        call_kwargs["temperature"] = config.temperature
    if config.extra_body:
        call_kwargs["extra_body"] = config.extra_body
    if config.api_base:
        call_kwargs["api_base"] = config.api_base
        call_kwargs["api_key"] = config.api_key
        if "/" not in model:
            # Route through the LiteLLM proxy; it resolves the actual provider
            model = f"litellm_proxy/{model}"
    # Adjustments (dropping an unsupported param, falling back to json_object)
    # don't consume transient-retry budget; each can happen at most once
    transient_failures = 0
    async with semaphore:
        while True:
            try:
                format_kwargs = response_format_kwargs(mode, schema_obj)
                completion = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    **call_kwargs,
                    **format_kwargs,
                )
                try:
                    cost = litellm.completion_cost(completion_response=completion)
                except Exception:
                    # Proxy/local models often have no litellm pricing entry
                    cost = 0.0
                choice = completion.choices[0]
                content = choice.message.content or ""
                error = rejection_error
                if not content:
                    finish = choice.finish_reason
                    print(f"Empty response — finish_reason={finish}")
                    error = f"empty_response: finish_reason={finish}; {error}"
                return {
                    "content": content,
                    "structured_mode_used": mode,
                    "schema_rejected": schema_rejected,
                    "cost": cost,
                    "error": error,
                }
            except litellm.BadRequestError as e:
                # Unsupported sampling param (e.g. gpt-5.x rejects temperature):
                # drop it and retry — this is NOT a schema rejection.
                if "temperature" in str(e).lower() and "temperature" in call_kwargs:
                    del call_kwargs["temperature"]
                    continue
                # Provider rejected the request — for json_schema mode this is
                # (almost always) the schema itself. Record it as a distinct
                # outcome, then fall back to plain JSON mode.
                if mode == "json_schema":
                    schema_rejected = True
                    rejection_error = f"schema_rejected: {e}"
                    mode = "json_object"
                    continue
                return {
                    "content": "",
                    "structured_mode_used": mode,
                    "schema_rejected": schema_rejected,
                    "cost": 0.0,
                    "error": f"BadRequestError: {e}",
                }
            except Exception as e:
                transient_failures += 1
                if transient_failures >= 4:
                    print(f"Failed after {transient_failures} attempts: {e}")
                    return {
                        "content": "",
                        "structured_mode_used": mode,
                        "schema_rejected": schema_rejected,
                        "cost": 0.0,
                        "error": str(e),
                    }
                await asyncio.sleep(2**transient_failures)


def load_done_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                done.add(record["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def aggregate(records: list[dict], metric_keys: list[str]) -> dict[str, dict]:
    """Means per group: overall, each complexity bucket, and the Q3 tails."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups["overall"].append(r)
        groups[str(r["complexity_bucket"])].append(r)
        depth = r.get("schema_depth")
        keys = r.get("schema_keys")
        if depth is not None and depth >= DEPTH_THRESHOLD:
            groups[DEPTH_GROUP].append(r)
        if keys is not None and keys >= KEYS_THRESHOLD:
            groups[KEYS_GROUP].append(r)

    summary = {}
    for name, rows in groups.items():
        entry = {"n": len(rows), "total_cost": sum(r.get("cost", 0.0) for r in rows)}
        for k in metric_keys:
            entry[k] = sum(r[k] for r in rows) / len(rows)
        rejected_count = sum(1 for r in rows if r.get("schema_rejected"))
        fallback_count = sum(
            1 for r in rows if r["structured_mode_used"] != "json_schema"
        )
        entry["schema_rejected_rate"] = rejected_count / len(rows)
        entry["fallback_rate"] = fallback_count / len(rows)
        summary[name] = entry
    return summary


def write_bucket_table(summary: dict, model_name: str, path: Path) -> None:
    cols = [
        "n",
        "schema_rejected_rate",
        "is_valid_json",
        "is_schema_compliant",
        "key_f1",
        "value_score",
    ]
    tail_groups = [UNKNOWN_BUCKET, DEPTH_GROUP, KEYS_GROUP]
    order = ["overall"] + [g for g in BUCKET_LABELS + tail_groups if g in summary]
    lines = [
        f"# {model_name} by schema complexity bucket",
        "",
        "| bucket | " + " | ".join(cols) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for bucket in order:
        entry = summary[bucket]
        cells = [str(entry[c]) if c == "n" else f"{entry[c]:.4f}" for c in cols]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")


def prepare_rows(
    config: FrontierEvalConfig,
) -> tuple[list[int], dict[int, dict], dict[int, str], Dataset]:
    ds = load_dataset(FT_REPO, split="test", revision=REVISION)
    assert isinstance(ds, Dataset)

    print("Loading schema stats from raw dataset...")
    stats_by_schema = load_schema_stats()
    empty = {"complexity": None, "depth": None, "keys": None}
    stats = {i: stats_by_schema.get(row["schema"], empty) for i, row in enumerate(ds)}
    unmatched = sum(1 for s in stats.values() if s["complexity"] is None)
    if unmatched:
        print(f"Warning: {unmatched} test schemas not found in raw dataset")

    if config.gold:
        with open(config.gold) as f:
            indices = sorted(json.loads(line)["test_idx"] for line in f)
        print(f"Restricting to {len(indices)} gold rows from {config.gold}")
    else:
        indices = (
            length_filter_indices(ds) if config.length_filter else list(range(len(ds)))
        )
    buckets = {i: bucket_for_score(s["complexity"]) for i, s in stats.items()}
    indices = stratified_sample(indices, buckets, config.sample, config.seed)
    deep_count = sum(1 for i in indices if (stats[i]["depth"] or 0) >= DEPTH_THRESHOLD)
    wide_count = sum(1 for i in indices if (stats[i]["keys"] or 0) >= KEYS_THRESHOLD)
    print(
        f"Evaluating {len(indices)} samples "
        f"({DEPTH_GROUP}: {deep_count}, {KEYS_GROUP}: {wide_count})"
    )
    return indices, stats, buckets, ds


async def run_eval(config: FrontierEvalConfig):
    indices, stats, buckets, ds = prepare_rows(config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", config.model_name)
    if config.gold:
        safe_model += "_gold"  # never clobber a full-test-set run
    jsonl_path = RESULTS_DIR / f"litellm_{safe_model}.jsonl"

    if config.resume:
        done = load_done_indices(jsonl_path)
        indices = [i for i in indices if i not in done]
        print(f"Resuming: {len(done)} already done, {len(indices)} remaining")
    else:
        open(jsonl_path, "w").close()  # fresh run: truncate any previous results

    semaphore = asyncio.Semaphore(config.concurrency)

    async def run_one(idx: int) -> tuple[int, dict]:
        row = ds[idx]
        prompt = prompts.build(row["schema"], row["content"])
        messages = [{"role": "user", "content": prompt}]
        schema_obj = json.loads(row["schema"])
        result = await call_model(semaphore, messages, schema_obj, config)
        return idx, result

    tasks = [asyncio.ensure_future(run_one(idx)) for idx in indices]

    # Write each record as soon as its call finishes so an interrupted run
    # keeps its progress and can continue with --resume
    with open(jsonl_path, "a") as f:
        for future in tqdm_asyncio.as_completed(
            tasks, total=len(tasks), desc=f"{config.model_name} inference"
        ):
            idx, result = await future
            row = ds[idx]
            schema_obj = json.loads(row["schema"])
            ground_truth = json.loads(row["response"])
            response = result["content"]
            clean_response = extract_json(response) if response else ""
            sample_metrics = M.run_all(clean_response, schema_obj, ground_truth)
            metric_values = {k: float(v) for k, v in sample_metrics.items()}

            record = {
                "idx": idx,
                "model": config.model_name,
                "complexity_score": stats[idx]["complexity"],
                "complexity_bucket": buckets[idx],
                "schema_depth": stats[idx]["depth"],
                "schema_keys": stats[idx]["keys"],
                "structured_mode_used": result["structured_mode_used"],
                "schema_rejected": result["schema_rejected"],
                "cost": result["cost"],
                "api_error": result["error"],
                "schema": row["schema"],
                "content": row["content"],
                "ground_truth": row["response"],
                "model_response": response,
                "clean_response": clean_response,
                **metric_values,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
    print(f"Saved {len(indices)} rows to {jsonl_path}")

    # Aggregate over everything in the file (includes resumed rows)
    with open(jsonl_path) as f:
        all_records = [json.loads(line) for line in f]
    if not all_records:
        print("No records to aggregate")
        return

    summary = aggregate(all_records, METRIC_KEYS)

    summary_path = RESULTS_DIR / f"litellm_{safe_model}_summary.json"
    config_dict = {**asdict(config), "api_key": "<redacted>"}
    summary_payload = {"config": config_dict, **summary}
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"Saved {summary_path}")

    TABLES_DIR.mkdir(exist_ok=True)
    table_path = TABLES_DIR / f"frontier_{safe_model}_by_complexity.md"
    write_bucket_table(summary, config.model_name, table_path)

    print(f"\n{'=' * 60}")
    for group, entry in summary.items():
        print(f"{group} (n={entry['n']}, cost=${entry['total_cost']:.2f}):")
        for k in [*METRIC_KEYS, "schema_rejected_rate", "fallback_rate"]:
            print(f"  {k}: {entry[k]:.4f}")
    print(f"{'=' * 60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--structured-mode",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Stratified sample size (0 = full set)"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON dict forwarded in the request body, e.g. '{\"repetition_penalty\": 1.1}'",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; omitted from the request by default",
    )
    parser.add_argument(
        "--no-length-filter",
        action="store_true",
        help="Skip the 8192-token prompt filter used by the vLLM eval",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gold",
        default=None,
        help="human_benchmark gold.jsonl: evaluate only its test_idx rows",
    )
    args = parser.parse_args()

    config = FrontierEvalConfig(
        model_name=args.model,
        structured_mode=args.structured_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
        api_base=os.environ.get("API_BASE"),
        api_key=os.environ.get("API_KEY"),
        sample=args.sample,
        concurrency=args.concurrency,
        seed=args.seed,
        length_filter=not args.no_length_filter,
        resume=args.resume,
        gold=args.gold,
    )
    print(f"Evaluating {config.model_name} via LiteLLM ({config.structured_mode})")
    asyncio.run(run_eval(config))


if __name__ == "__main__":
    main()
