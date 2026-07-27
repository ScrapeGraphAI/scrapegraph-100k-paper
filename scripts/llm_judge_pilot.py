import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import regeneration_comparison as rc  # noqa: E402

from modelling.metrics import (  # noqa: E402
    _flatten_dict,
    parse_json_remove_duplicates,
)

RAW_REPO = "scrapegraphai/scrapegraphai-100k"
REVISION = "v1.0"

OUT_DIR = Path(".data/llm_judge_pilot")
TABLES_DIR = Path("tables")

UNITS_PATH = OUT_DIR / "units.jsonl"
STRATA_PATH = OUT_DIR / "strata.json"
JUDGMENTS_PATH = OUT_DIR / "judgments.jsonl"

# mirror modelling/preprocess.py: one content chunk = 4096 tokens ~ 14,336 chars
ONE_CHUNK_CHARS = int(rc.CHUNK_SIZE_TOKENS * rc.CHARS_PER_TOKEN)
MAX_CHARS = rc.MAX_CHARS  # content 50k / schema 10k / response 10k, as in preprocess

# domain-tier allocation of the main sample (head is capped so templated
# amazon/publisher pages don't dominate; tail is where reviewer-visible
# low-quality examples most likely live)
TIER_SHARES = {"head": 0.25, "torso": 0.30, "tail": 0.45}
HEAD_RANKS = 10
TORSO_RANKS = 100

MAX_LEAVES = 60  # leaves shown to the judge per example
LEAF_VALUE_CHARS = 300

TIERS = ["gold", "silver", "bronze", "reject"]
TIER_ORDER = {t: i for i, t in enumerate(reversed(TIERS))}  # reject=0 .. gold=3
FILLED_VERDICTS = ["supported", "partially_supported", "hallucinated", "wrong_field"]
MISSING_VERDICTS = ["correctly_missing", "missed_in_content"]

JUDGE_RESPONSE_SCHEMA = {
    "name": "extraction_quality_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "leaves": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": FILLED_VERDICTS + MISSING_VERDICTS,
                        },
                    },
                    "required": ["key", "verdict"],
                    "additionalProperties": False,
                },
            },
            "tier": {"type": "string", "enum": TIERS},
            "justification": {"type": "string"},
        },
        "required": ["leaves", "tier", "justification"],
        "additionalProperties": False,
    },
}

JUDGE_PROMPT = """You are auditing the quality of a structured web-data extraction.

Below is (1) the page content the extraction model saw, (2) the JSON schema it had to follow, and (3) the extraction it produced, flattened to leaf fields.

For EVERY leaf field listed, judge the extracted value strictly against the page content:
- supported: value is correct and directly supported by the content
- partially_supported: value is grounded in the content but incomplete, imprecise, or partly wrong
- hallucinated: value does not appear in and cannot reasonably be inferred from the content
- wrong_field: value appears in the content but belongs in a different field or at the wrong granularity
- correctly_missing: the field holds a missing-marker (null, "NA", "", empty list, "Not specified", ...) and the content indeed has no value for it
- missed_in_content: the field holds a missing-marker but the content DOES contain a value for it

Then assign one overall tier:
- gold: all or nearly all filled fields supported; no hallucinations; nothing important missed
- silver: minor issues — a few partially supported values, or one minor hallucinated or missed field
- bronze: substantial issues — several hallucinated, wrong, or missed fields, but the extraction is still mostly usable
- reject: the extraction is largely wrong, hallucinated, or unrelated to the content

Judge ONLY against the provided content. Copy each leaf's key string exactly as given.

PAGE CONTENT:
{content}

JSON SCHEMA:
{schema}

EXTRACTION (flattened leaf fields):
{leaves}
"""


# ---------------------------------------------------------------- sampling


def flatten_response(response_str: str) -> dict | None:
    try:
        parsed = parse_json_remove_duplicates(response_str)
        normalized = rc.normalize_root(parsed)
        return _flatten_dict(normalized)
    except Exception:
        return None


# raw v1.0 rows with strata columns, restricted to rows the finetuning
# preprocess char caps would keep and a parseable response — exclusion
# counts are printed so the report can state them
def load_raw_eligible() -> pd.DataFrame:
    import tldextract

    path = hf_hub_download(
        RAW_REPO, "data/train.parquet", repo_type="dataset", revision=REVISION
    )
    df = pq.read_table(
        path, columns=["source", "schema", "content", "response"]
    ).to_pandas()
    df["row_idx"] = df.index
    n_total = len(df)

    caps_ok = (
        df["content"].str.len().le(MAX_CHARS["content"])
        & df["schema"].str.len().le(MAX_CHARS["schema"])
        & df["response"].str.len().le(MAX_CHARS["response"])
        & df["content"].notna()
        & df["schema"].notna()
        & df["response"].notna()
    )
    df = df[caps_ok]
    print(f"Raw rows: {n_total:,}; within preprocess char caps: {len(df):,}")

    parseable = df["response"].map(lambda r: flatten_response(r) is not None)
    df = df[parseable]
    print(f"With parseable JSON response: {len(df):,}")

    extract = tldextract.TLDExtract(suffix_list_urls=())

    def root_domain(source) -> str | None:
        if not isinstance(source, str) or not source.strip():
            return None
        url = source.strip()
        if "://" not in url:
            url = "https://" + url
        raw_host = urlparse(url).hostname or ""
        host = raw_host.lower().rstrip(".")
        if not host:
            return None
        ext = extract(host)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else None

    df["root_domain"] = df["source"].map(root_domain)
    ranks = df["root_domain"].value_counts()
    head = set(ranks.index[:HEAD_RANKS])
    torso = set(ranks.index[HEAD_RANKS:TORSO_RANKS])
    df["domain_tier"] = [
        "head" if d in head else "torso" if d in torso else "tail"
        for d in df["root_domain"]
    ]
    df["long_page"] = df["content"].str.len() > ONE_CHUNK_CHARS
    return df


def stratified_main_sample(df: pd.DataFrame, n_main: int, seed: int) -> pd.DataFrame:
    picks = []
    for tier, share in TIER_SHARES.items():
        want = round(n_main * share)
        short = df[(df["domain_tier"] == tier) & ~df["long_page"]]
        long = df[(df["domain_tier"] == tier) & df["long_page"]]
        k_short = min(want // 2, len(short))
        k_long = min(want - k_short, len(long))
        k_short = min(want - k_long, len(short))  # top up if long is scarce
        picks.append(short.sample(n=k_short, random_state=seed))
        picks.append(long.sample(n=k_long, random_state=seed))
    sample = pd.concat(picks).sample(frac=1, random_state=seed)  # shuffle
    return sample


# reuse the D2 alignment; each pair yields two judge units — the original
# page-level target judged against the full page, and the regenerated target
# judged against the chunk the regeneration model saw
def sample_pairs(n_pairs: int, seed: int) -> list[dict]:
    index, by_schema, raw_rows = rc.build_raw_index()
    ft = rc.load_finetuning()

    aligned = []  # (ft row position, raw row position, full_page)
    for i, (schema, chunk) in enumerate(zip(ft["schema"], ft["content"], strict=True)):
        positions = index.get(rc.chunk_key(schema, chunk))
        if not positions:
            probe = chunk[: rc.PREFIX_PROBE_CHARS]
            positions = [
                p for p in by_schema.get(schema, []) if probe in raw_rows[p][0]
            ]
        if not positions:
            continue
        distinct_responses = {raw_rows[p][1] for p in positions}
        if len(distinct_responses) > 1:
            continue
        full_page = all(raw_rows[p][0] == chunk for p in positions)
        aligned.append((i, positions[0], full_page))
    print(f"Aligned pairs available: {len(aligned):,}")

    rng = random.Random(seed)
    chosen = rng.sample(aligned, min(n_pairs, len(aligned)))

    pairs = []
    for pair_id, (ft_pos, raw_pos, full_page) in enumerate(chosen):
        page_content, orig_response = raw_rows[raw_pos]
        pairs.append(
            {
                "pair_id": pair_id,
                "schema": ft["schema"].iloc[ft_pos],
                "page_content": page_content,
                "chunk_content": ft["content"].iloc[ft_pos],
                "orig_response": orig_response,
                "regen_response": ft["response"].iloc[ft_pos],
                "full_page": full_page,
            }
        )
    return pairs


def stage_sample(args) -> None:
    if UNITS_PATH.exists():
        print(f"{UNITS_PATH} exists — skipping sampling (delete it to resample)")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_eligible()
    main = stratified_main_sample(df, args.n_main, args.seed)

    population = df.groupby(["domain_tier", "long_page"]).size()
    sampled = main.groupby(["domain_tier", "long_page"]).size()
    strata_table = pd.concat(
        {"population": population, "sampled": sampled}, axis=1
    ).fillna(0)
    strata = strata_table.reset_index().to_dict(orient="records")
    STRATA_PATH.write_text(json.dumps(strata, indent=2))
    print(f"Main sample: {len(main)} rows")
    print(strata_table)

    units = []
    for row in main.to_dict(orient="records"):
        units.append(
            {
                "unit_id": f"main:{row['row_idx']}",
                "kind": "main",
                "row_idx": int(row["row_idx"]),
                "source": row["source"],
                "root_domain": row["root_domain"],
                "domain_tier": row["domain_tier"],
                "long_page": bool(row["long_page"]),
                "schema": row["schema"],
                "content": row["content"],
                "response": row["response"],
            }
        )

    for pair in sample_pairs(args.n_pairs, args.seed):
        base = {
            "pair_id": pair["pair_id"],
            "schema": pair["schema"],
            "full_page": pair["full_page"],
        }
        units.append(
            base
            | {
                "unit_id": f"pair:{pair['pair_id']}:orig",
                "kind": "pair_orig",
                "content": pair["page_content"],
                "response": pair["orig_response"],
            }
        )
        units.append(
            base
            | {
                "unit_id": f"pair:{pair['pair_id']}:regen",
                "kind": "pair_regen",
                "content": pair["chunk_content"],
                "response": pair["regen_response"],
            }
        )

    with open(UNITS_PATH, "w") as f:
        for unit in units:
            f.write(json.dumps(unit) + "\n")
    print(f"Saved {len(units)} judge units to {UNITS_PATH}")


# ---------------------------------------------------------------- judging


def leaf_lines(flat: dict) -> str:
    lines = []
    for key, value in list(flat.items())[:MAX_LEAVES]:
        # _flatten_dict marks empty-dict leaves with a bare object() sentinel
        # (EMPTY_DICT_SENTINEL) — render those, and anything else json can't
        # serialize, as {}
        rendered = json.dumps(value, ensure_ascii=False, default=lambda o: {})
        if len(rendered) > LEAF_VALUE_CHARS:
            rendered = rendered[:LEAF_VALUE_CHARS] + "…"
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines)


def build_judge_prompt(unit: dict) -> str:
    flat = flatten_response(unit["response"]) or {}
    return JUDGE_PROMPT.format(
        content=unit["content"], schema=unit["schema"], leaves=leaf_lines(flat)
    )


# LiteLLM proxy from .env (API_KEY/API_BASE) when configured, else the
# standard OPENAI_* environment
def make_client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if os.getenv("API_KEY") and os.getenv("API_BASE"):
        return OpenAI(
            api_key=os.environ["API_KEY"], base_url=os.environ["API_BASE"], timeout=300
        )
    return OpenAI(timeout=300)


@retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(5))
def call_judge(client: OpenAI, model: str, prompt: str) -> dict:
    kwargs = {}
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = "low"
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": JUDGE_RESPONSE_SCHEMA,
        },
        **kwargs,
    )  # ty: ignore[no-matching-overload]
    return json.loads(completion.choices[0].message.content)


def load_units() -> list[dict]:
    if not UNITS_PATH.exists():
        sys.exit(f"{UNITS_PATH} not found — run the sample stage first")
    return [json.loads(line) for line in open(UNITS_PATH)]


def load_judgments() -> list[dict]:
    if not JUDGMENTS_PATH.exists():
        return []
    records, n_corrupt = [], 0
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # truncated line from a killed run
                n_corrupt += 1
    if n_corrupt:
        print(
            f"Skipped {n_corrupt} corrupt line(s) in {JUDGMENTS_PATH} "
            "(interrupted write); those units will be re-judged"
        )
    return records


def stage_judge(args) -> None:
    units = load_units()
    # error records don't count as done — they are retried on the next run
    done = {
        (j["unit_id"], j["judge_model"]) for j in load_judgments() if "error" not in j
    }

    work = [(u, args.judge_model) for u in units]
    if args.second_judge:
        stability_units = [u for u in units if u["kind"] == "main"][: args.stability_n]
        work += [(u, args.second_judge) for u in stability_units]
    work = [(u, m) for u, m in work if (u["unit_id"], m) not in done]
    if args.limit:
        work = work[: args.limit]
    if not work:
        print("Nothing left to judge")
        return

    input_chars = sum(len(u["content"]) + len(u["schema"]) for u, _ in work)
    est_tokens = input_chars / 3.5
    print(
        f"Judging {len(work)} units ({len(done)} already cached); "
        f"~{est_tokens / 1e6:.1f}M estimated input tokens"
    )

    client = make_client()

    def run(unit: dict, model: str) -> dict:
        record = {"unit_id": unit["unit_id"], "judge_model": model}
        try:  # nothing in here may kill the run — errors surface in analyze
            record.update(call_judge(client, model, build_judge_prompt(unit)))
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        return record

    with open(JUDGMENTS_PATH, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, u, m) for u, m in work]
        n_errors = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            record = future.result()
            n_errors += "error" in record
            # only this thread writes, so no locking needed
            out.write(json.dumps(record) + "\n")
            out.flush()
    print(
        f"Done: {len(work) - n_errors} judged, {n_errors} errors "
        f"(cached in {JUDGMENTS_PATH})"
    )


# ---------------------------------------------------------------- analysis


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    pe = sum(counts_a[t] * counts_b[t] for t in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


# leaf-verdict rates, matched back to the flattened response so filled and
# missing-marker leaves are separated by OUR convention (rc.is_missing),
# not the judge's
def leaf_rates(units_by_id: dict, judgments: list[dict]) -> dict:
    filled = Counter()
    missing = Counter()
    n_unmatched = 0
    units_with_hallucination = 0
    for j in judgments:
        unit = units_by_id[j["unit_id"]]
        flat = flatten_response(unit["response"]) or {}
        any_hallucinated = False
        seen_keys = set()
        for leaf in j.get("leaves", []):
            if leaf["key"] not in flat:
                n_unmatched += 1
                continue
            if leaf["key"] in seen_keys:  # judge repeated a key
                continue
            seen_keys.add(leaf["key"])
            if rc.is_missing(flat[leaf["key"]]):
                missing[leaf["verdict"]] += 1
            else:
                filled[leaf["verdict"]] += 1
            any_hallucinated |= leaf["verdict"] == "hallucinated"
        units_with_hallucination += any_hallucinated

    n_filled, n_miss = sum(filled.values()), sum(missing.values())
    rates = {
        "n_filled_leaves": n_filled,
        "n_missing_leaves": n_miss,
        "n_unmatched_leaf_keys": n_unmatched,
    }
    for verdict in FILLED_VERDICTS:
        rates[f"filled_{verdict}_rate"] = round(filled[verdict] / max(n_filled, 1), 4)
    for verdict in MISSING_VERDICTS:
        rates[f"missing_{verdict}_rate"] = round(missing[verdict] / max(n_miss, 1), 4)
    # crossovers: judge used a missing-type verdict on a filled leaf or a
    # filled-type verdict on a missing-marker leaf — reported so each verdict
    # family sums to 1
    rates["filled_crossover_rate"] = round(
        sum(filled[v] for v in MISSING_VERDICTS) / max(n_filled, 1), 4
    )
    rates["missing_crossover_rate"] = round(
        sum(missing[v] for v in FILLED_VERDICTS) / max(n_miss, 1), 4
    )
    rates["units_with_any_hallucination_rate"] = round(
        units_with_hallucination / max(len(judgments), 1), 4
    )
    return rates


def tier_shares(judgments: list[dict]) -> dict[str, float]:
    counts = Counter(j["tier"] for j in judgments)
    n = max(sum(counts.values()), 1)
    return {t: counts[t] / n for t in TIERS}


# original vs regenerated targets on the D2-aligned paired slice
def analyze_pairs(metrics: dict, units_by_id: dict, primary: list[dict]) -> None:
    orig, regen = {}, {}  # pair_id -> judgment, per side
    for j in primary:
        unit = units_by_id[j["unit_id"]]
        if unit["kind"] == "pair_orig":
            orig[unit["pair_id"]] = j
        elif unit["kind"] == "pair_regen":
            regen[unit["pair_id"]] = j
    both = sorted(set(orig) & set(regen))
    metrics["pairs_judged"] = len(both)
    if not both:
        return
    for label, side in [("orig", orig), ("regen", regen)]:
        judged = [side[p] for p in both]
        for tier, share in tier_shares(judged).items():
            metrics[f"pair_{label}_tier_{tier}_rate"] = round(share, 4)
        rates = leaf_rates(units_by_id, judged)
        metrics.update(
            {
                f"pair_{label}_{k}": v
                for k, v in rates.items()
                if k.startswith(("filled_", "missing_", "units_"))
            }
        )
    deltas = [TIER_ORDER[regen[p]["tier"]] - TIER_ORDER[orig[p]["tier"]] for p in both]
    metrics["pair_regen_better_rate"] = round(
        sum(d > 0 for d in deltas) / len(deltas), 4
    )
    metrics["pair_regen_worse_rate"] = round(
        sum(d < 0 for d in deltas) / len(deltas), 4
    )
    metrics["pair_same_tier_rate"] = round(sum(d == 0 for d in deltas) / len(deltas), 4)


def analyze_stability(
    metrics: dict, main: list[dict], all_judgments: list[dict], second_judge: str | None
) -> None:
    if not second_judge:  # fall back to any other judge present in the cache
        others = Counter(
            j["judge_model"]
            for j in all_judgments
            if j["judge_model"] != metrics["judge_model"] and "error" not in j
        )
        second_judge = others.most_common(1)[0][0] if others else None
    if not second_judge:
        return
    metrics["stability_judge"] = second_judge
    second = {
        j["unit_id"]: j
        for j in all_judgments
        if j["judge_model"] == second_judge and "error" not in j
    }
    main_by_id = {j["unit_id"]: j for j in main}
    overlap = [uid for uid in main_by_id if uid in second]
    if not overlap:
        return
    a = [main_by_id[uid]["tier"] for uid in overlap]
    b = [second[uid]["tier"] for uid in overlap]
    metrics["stability_n"] = len(overlap)
    metrics["stability_agreement"] = round(
        sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 4
    )
    metrics["stability_kappa"] = round(cohens_kappa(a, b), 4)


def stage_analyze(args) -> None:
    units = load_units()
    units_by_id = {u["unit_id"]: u for u in units}
    all_judgments = load_judgments()
    primary = [
        j
        for j in all_judgments
        if j["judge_model"] == args.judge_model and "error" not in j
    ]
    if not primary:
        sys.exit(f"No judgments from {args.judge_model} — run the judge stage first")
    # last judgment wins if a unit was ever re-judged
    latest_by_unit = {j["unit_id"]: j for j in primary}
    primary = list(latest_by_unit.values())

    main = [j for j in primary if units_by_id[j["unit_id"]]["kind"] == "main"]
    # only units that never got a successful judgment count as errors
    error_ids = {
        j["unit_id"]
        for j in all_judgments
        if j["judge_model"] == args.judge_model and "error" in j
    }
    judged_ids = {j["unit_id"] for j in primary}
    n_errors = len(error_ids - judged_ids)

    metrics: dict = {
        "judge_model": args.judge_model,
        "units_judged": len(primary),
        "main_judged": len(main),
        "judge_error_count": n_errors,
    }

    # --- main sample: tier distribution, raw and reweighted to the dataset ---
    for tier, share in tier_shares(main).items():
        metrics[f"main_tier_{tier}_rate"] = round(share, 4)

    strata_records = json.loads(STRATA_PATH.read_text())
    strata = {(s["domain_tier"], s["long_page"]): s for s in strata_records}
    total_population = sum(s["population"] for s in strata.values())
    stratum_of = {
        u["unit_id"]: (u["domain_tier"], u["long_page"])
        for u in units
        if u["kind"] == "main"
    }
    weighted = Counter()
    for stratum_key, s in strata.items():
        cell = [j for j in main if stratum_of[j["unit_id"]] == stratum_key]
        if not cell:
            continue
        weight = s["population"] / total_population
        for tier, share in tier_shares(cell).items():
            weighted[tier] += weight * share
    for tier in TIERS:
        metrics[f"main_tier_{tier}_rate_weighted"] = round(weighted[tier], 4)

    # --- slices ---
    for name, predicate in [
        ("head", lambda u: u["domain_tier"] == "head"),
        ("torso", lambda u: u["domain_tier"] == "torso"),
        ("tail", lambda u: u["domain_tier"] == "tail"),
        ("short_page", lambda u: not u["long_page"]),
        ("long_page", lambda u: u["long_page"]),
    ]:
        cell = [j for j in main if predicate(units_by_id[j["unit_id"]])]
        metrics[f"slice_{name}_n"] = len(cell)
        shares = tier_shares(cell)
        metrics[f"slice_{name}_gold_rate"] = round(shares["gold"], 4)
        metrics[f"slice_{name}_gold_or_silver_rate"] = round(
            shares["gold"] + shares["silver"], 4
        )
        metrics[f"slice_{name}_reject_rate"] = round(shares["reject"], 4)

    # --- leaf-level rates on the main sample ---
    metrics.update({f"main_{k}": v for k, v in leaf_rates(units_by_id, main).items()})

    analyze_pairs(metrics, units_by_id, primary)
    analyze_stability(metrics, main, all_judgments, args.second_judge)

    print("\n--- E: LLM-judge semantic quality pilot ---")
    for k, v in metrics.items():
        print(f"{k:>45}: {v}")

    TABLES_DIR.mkdir(exist_ok=True)
    pd.DataFrame(metrics.items(), columns=["metric", "value"]).to_csv(
        TABLES_DIR / "llm_judge_pilot.csv", index=False
    )
    print(f"\nSaved {TABLES_DIR / 'llm_judge_pilot.csv'}")


# ---------------------------------------------------------------- entrypoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", nargs="?", default="all", choices=["sample", "judge", "analyze", "all"]
    )
    parser.add_argument("--n-main", type=int, default=1000)
    parser.add_argument("--n-pairs", type=int, default=250)
    parser.add_argument("--judge-model", default="gpt-5.6-terra-fiit")
    parser.add_argument(
        "--second-judge",
        default=None,
        help="second judge model for the stability subsample",
    )
    parser.add_argument("--stability-n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge at most N units this run (smoke test)",
    )
    parser.add_argument("--seed", type=int, default=rc.SEED)
    args = parser.parse_args()

    if args.stage in ("sample", "all"):
        stage_sample(args)
    if args.stage in ("judge", "all"):
        stage_judge(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)


if __name__ == "__main__":
    main()
