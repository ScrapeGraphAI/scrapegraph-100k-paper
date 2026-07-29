import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import orjson
import pandas as pd
import pyarrow.parquet as pq
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelling.metrics import (  # noqa: E402
    _flatten_dict,
    json_validator,
    magic_metric,
    parse_json_remove_duplicates,
)

RAW_REPO = "scrapegraphai/scrapegraphai-100k"
FT_REPO = "scrapegraphai/scrapegraph-100k-finetuning"
REVISION = "v1.0"

# mirror modelling/preprocess.py exactly so chunks reproduce byte-for-byte
CHARS_PER_TOKEN = 3.5
CHUNK_SIZE_TOKENS = 4096
OVERLAP_TOKENS = 128
MAX_CHARS = {"content": 50_000, "schema": 10_000, "response": 10_000}

SAMPLE_N = 5_000  # pairs for the BLEU-based soft metrics
SEED = 42

TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(exist_ok=True)


def chunk_in_chars(src: str, size: int, overlap: int) -> list[str]:
    if not src:
        return []
    step = size - overlap
    return [src[i : i + size] for i in range(0, len(src), step)]


def chunk_key(schema: str, chunk: str) -> bytes:
    key_material = schema.encode() + b"\x00" + chunk.encode()
    return hashlib.blake2b(key_material, digest_size=16).digest()


def build_raw_index() -> (
    tuple[dict[bytes, list[int]], dict[str, list[int]], list[tuple[str, str]]]
):
    # rows over the preprocess char caps could never have produced a chunk
    path = hf_hub_download(
        RAW_REPO, "data/train.parquet", repo_type="dataset", revision=REVISION
    )
    table = pq.read_table(path, columns=["schema", "content", "response"])
    print(f"Raw dataset: {table.num_rows:,} rows")

    size = int(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)
    overlap = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)

    index: dict[bytes, list[int]] = {}
    by_schema: dict[str, list[int]] = {}
    rows: list[tuple[str, str]] = []
    for batch in table.to_batches(max_chunksize=10_000):
        for schema, content, response in zip(
            batch["schema"].to_pylist(),
            batch["content"].to_pylist(),
            batch["response"].to_pylist(),
            strict=True,
        ):
            passes_char_filters = (
                isinstance(content, str)
                and len(content) <= MAX_CHARS["content"]
                and isinstance(schema, str)
                and len(schema) <= MAX_CHARS["schema"]
                and isinstance(response, str)
                and len(response) <= MAX_CHARS["response"]
            )
            if not passes_char_filters:
                continue
            pos = len(rows)
            rows.append((content, response))
            by_schema.setdefault(schema, []).append(pos)
            for chunk in chunk_in_chars(content, size, overlap):
                key = chunk_key(schema, chunk)
                index.setdefault(key, []).append(pos)

    print(f"Raw rows passing preprocess char filters: {len(rows):,}")
    print(f"Distinct (schema, chunk) keys: {len(index):,}")
    return index, by_schema, rows


def load_finetuning() -> pd.DataFrame:
    ds = load_dataset(FT_REPO, revision=REVISION)
    assert isinstance(ds, DatasetDict)
    frames = []
    for split, split_ds in ds.items():
        df = split_ds.to_pandas()
        assert isinstance(df, pd.DataFrame)
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    split_sizes = ", ".join(f"{s}: {len(split_ds):,}" for s, split_ds in ds.items())
    print(f"Finetuning dataset: {len(out):,} rows ({split_sizes})")
    return out


def leaf_stats(flat: dict) -> tuple[int, int, int]:
    # nulls counted separately from missing markers: the original targets write
    # 'NA' where the regenerated ones write null/""/"Not specified"
    n_null = n_missing = 0
    for v in flat.values():
        is_all_null_list = isinstance(v, list) and v and all(x is None for x in v)
        if v is None or is_all_null_list:
            n_null += 1
        if is_missing(v):
            n_missing += 1
    return len(flat), n_null, n_missing


def canonical(v) -> bytes:
    # metrics._flatten_dict marks empty dicts with a bare object() sentinel
    # that orjson cannot serialize; any such value canonicalizes to a fixed
    # marker so identical sentinels compare equal instead of crashing
    return orjson.dumps(
        v, option=orjson.OPT_SORT_KEYS, default=lambda _: "<unserializable>"
    )


MISSING_STRINGS = {"", "na", "n/a", "none", "null", "unknown", "-"}
# verbose absence phrasings gpt-5-nano uses instead of null/NA
MISSING_PREFIXES = (
    "not specified",
    "not provided",
    "not disclosed",
    "not available",
    "not mentioned",
    "not applicable",
    "not found",
    "not given",
    "not listed",
    "no information",
)


def is_missing(v) -> bool:
    # 'no information' under any convention seen in the data
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in MISSING_STRINGS or s.startswith(MISSING_PREFIXES)
    if isinstance(v, list):
        return len(v) == 0 or all(is_missing(x) for x in v)
    return False


def classify_change(o, r) -> str:
    if is_missing(o) and is_missing(r):
        return "chg_missing_convention"
    if is_missing(o):
        return "chg_orig_missing_regen_content"
    if is_missing(r):
        return "chg_orig_content_regen_missing"
    if (
        isinstance(o, str)
        and isinstance(r, str)
        and o.strip().casefold() == r.strip().casefold()
    ):
        return "chg_case_whitespace_only"
    both_numeric = (
        isinstance(o, (int, float))
        and not isinstance(o, bool)
        and isinstance(r, (int, float))
        and not isinstance(r, bool)
    )
    if both_numeric and math.isclose(o, r, rel_tol=1e-9, abs_tol=1e-12):
        return "chg_numeric_format"  # e.g. 1 vs 1.0
    return "chg_content_rewrite"


def normalize_root(parsed) -> dict:
    return parsed if isinstance(parsed, dict) else {"[root]": parsed}


def compare_pair(schema: dict, orig_str: str, regen_str: str) -> dict:
    orig_validation = json_validator(orig_str, schema)
    regen_validation = json_validator(regen_str, schema)
    result = {
        "orig_valid_json": orig_validation["is_valid"],
        "orig_compliant": orig_validation["is_compliant"],
        "regen_valid_json": regen_validation["is_valid"],
        "regen_compliant": regen_validation["is_compliant"],
        "orig_chars": len(orig_str),
        "regen_chars": len(regen_str),
    }
    if not (orig_validation["is_valid"] and regen_validation["is_valid"]):
        return result

    orig_parsed = parse_json_remove_duplicates(orig_str)
    regen_parsed = parse_json_remove_duplicates(regen_str)
    orig = normalize_root(orig_parsed)
    regen = normalize_root(regen_parsed)
    flat_orig = _flatten_dict(orig)
    flat_regen = _flatten_dict(regen)

    orig_keys, regen_keys = set(flat_orig), set(flat_regen)
    shared = orig_keys & regen_keys
    union = orig_keys | regen_keys
    result["jaccard"] = len(shared) / len(union) if union else 1.0
    result["identical_keys"] = orig_keys == regen_keys
    # top-level view: robust to the flattening quirk where an empty list
    # ("urls") and a filled list ("urls[*].url") share no flattened keys
    top_shared = set(orig) & set(regen)
    top_union = set(orig) | set(regen)
    result["top_jaccard"] = len(top_shared) / len(top_union) if top_union else 1.0
    result["n_keys_orig"], result["n_null_orig"], result["n_missing_orig"] = leaf_stats(
        flat_orig
    )
    result["n_keys_regen"], result["n_null_regen"], result["n_missing_regen"] = (
        leaf_stats(flat_regen)
    )

    n_changed = 0
    transitions = Counter()
    for k in shared:
        o, r = flat_orig[k], flat_regen[k]
        if canonical(o) != canonical(r):
            n_changed += 1
            transitions[classify_change(o, r)] += 1
    result["n_shared"] = len(shared)
    result["n_changed"] = n_changed
    result.update(transitions)
    return result


PREFIX_PROBE_CHARS = 1_000  # chunk prefix used for the substring fallback


def align(
    ft: pd.DataFrame, index, by_schema, raw_rows
) -> list[tuple[str, str, str, bool]]:
    # exact (schema, chunk) hash match first; prefix-substring fallback catches
    # pages whose content drifted between the intermediate parquet and v1.0.
    # A match is kept only if all candidate source pages agree on the response.
    pairs = []
    n_exact = n_prefix = n_unmatched = n_ambiguous = 0

    for schema, content, regen in zip(
        ft["schema"], ft["content"], ft["response"], strict=True
    ):
        positions = index.get(chunk_key(schema, content))
        if positions:
            match_type = "exact"
        else:
            probe = content[:PREFIX_PROBE_CHARS]
            positions = [
                p for p in by_schema.get(schema, []) if probe in raw_rows[p][0]
            ]
            match_type = "prefix"
        if not positions:
            n_unmatched += 1
            continue
        responses = {raw_rows[p][1] for p in positions}
        if len(responses) > 1:
            n_ambiguous += 1
            continue
        if match_type == "exact":
            n_exact += 1
        else:
            n_prefix += 1
        # full-page pairs: the chunk IS the whole page, so the regeneration
        # model saw everything the original annotation was based on and any
        # orig/regen difference is model-driven, not truncation-driven
        full_page = all(raw_rows[p][0] == content for p in positions)
        orig_response = responses.pop()
        pairs.append((schema, orig_response, regen, full_page))

    n_total = len(ft)
    print(
        f"\nAligned pairs: {len(pairs):,}/{n_total:,} ({len(pairs) / n_total:.1%}) "
        f"[exact: {n_exact:,}, prefix-fallback: {n_prefix:,}]; "
        f"unmatched: {n_unmatched:,}; "
        f"ambiguous source (conflicting originals): {n_ambiguous:,}"
    )
    return pairs


def compare_all(pairs: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    schema_cache: dict[str, dict | None] = {}  # None = unparseable schema
    records = []
    n_bad_schema = 0
    for schema_str, orig, regen, full_page in pairs:
        if schema_str not in schema_cache:
            try:
                schema_cache[schema_str] = json.loads(schema_str)
            except json.JSONDecodeError:
                schema_cache[schema_str] = None
        schema = schema_cache[schema_str]
        if schema is None:
            n_bad_schema += 1
            continue
        record = compare_pair(schema, orig, regen)
        record["full_page"] = full_page
        records.append(record)
    if n_bad_schema:
        print(f"Skipped {n_bad_schema} aligned pairs with unparseable schemas")
    return pd.DataFrame(records)


def soft_sample_metrics(pairs: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    # BLEU-based soft comparison: regenerated scored against original
    rng = random.Random(SEED)
    sample = rng.sample(pairs, min(SAMPLE_N, len(pairs)))
    soft = []
    for _schema_str, orig, regen, _ in sample:
        try:
            regen_parsed = parse_json_remove_duplicates(regen)
            orig_parsed = parse_json_remove_duplicates(orig)
            regen_tree = normalize_root(regen_parsed)
            orig_tree = normalize_root(orig_parsed)
            soft.append(magic_metric(regen_tree, orig_tree))
        except Exception:
            continue
    return pd.DataFrame(soft)


def main() -> None:
    index, by_schema, raw_rows = build_raw_index()
    ft = load_finetuning()
    pairs = align(ft, index, by_schema, raw_rows)
    n_total = len(ft)

    df = compare_all(pairs)
    n_pairs = len(df)
    soft_df = soft_sample_metrics(pairs)

    both_parsed = df.dropna(subset=["jaccard"])
    shared_total = int(both_parsed["n_shared"].sum())
    changed_total = int(both_parsed["n_changed"].sum())
    total_leaf_keys_orig = both_parsed["n_keys_orig"].sum()
    total_leaf_keys_regen = both_parsed["n_keys_regen"].sum()
    no_shared_value_change = both_parsed["n_changed"] == 0
    fully_identical = both_parsed["identical_keys"] & no_shared_value_change

    metrics = {
        "aligned_pairs": n_pairs,
        "alignment_rate": round(len(pairs) / n_total, 4),
        # validity delta
        "orig_valid_json_rate": round(df["orig_valid_json"].mean(), 4),
        "regen_valid_json_rate": round(df["regen_valid_json"].mean(), 4),
        "orig_schema_compliant_rate": round(df["orig_compliant"].mean(), 4),
        "regen_schema_compliant_rate": round(df["regen_compliant"].mean(), 4),
        # key overlap
        "mean_key_jaccard": round(both_parsed["jaccard"].mean(), 4),
        "median_key_jaccard": round(both_parsed["jaccard"].median(), 4),
        "mean_top_level_key_jaccard": round(both_parsed["top_jaccard"].mean(), 4),
        "identical_key_sets_rate": round(both_parsed["identical_keys"].mean(), 4),
        "zero_shared_key_pairs_rate": round((both_parsed["n_shared"] == 0).mean(), 4),
        "mean_keys_orig": round(both_parsed["n_keys_orig"].mean(), 2),
        "mean_keys_regen": round(both_parsed["n_keys_regen"].mean(), 2),
        # style / length shift
        "median_chars_orig": float(df["orig_chars"].median()),
        "median_chars_regen": float(df["regen_chars"].median()),
        "null_leaf_rate_orig": round(
            both_parsed["n_null_orig"].sum() / total_leaf_keys_orig, 4
        ),
        "null_leaf_rate_regen": round(
            both_parsed["n_null_regen"].sum() / total_leaf_keys_regen, 4
        ),
        # missing markers of any convention (null, 'NA', "", [], "Not specified")
        "missing_leaf_rate_orig": round(
            both_parsed["n_missing_orig"].sum() / total_leaf_keys_orig, 4
        ),
        "missing_leaf_rate_regen": round(
            both_parsed["n_missing_regen"].sum() / total_leaf_keys_regen, 4
        ),
        # value-change rate (shared keys, exact comparison)
        "value_change_rate": round(changed_total / shared_total, 4),
        # identical structure AND no shared-key value change
        "pairs_identical_rate": round(fully_identical.mean(), 4),
        "pairs_no_shared_value_change_rate": round(no_shared_value_change.mean(), 4),
    }
    # what the changed values actually are (rates over changed shared keys)
    for col in (
        "chg_missing_convention",
        "chg_orig_missing_regen_content",
        "chg_orig_content_regen_missing",
        "chg_case_whitespace_only",
        "chg_numeric_format",
        "chg_content_rewrite",
    ):
        if col in both_parsed:
            metrics[f"{col}_rate"] = round(
                both_parsed[col].sum() / max(changed_total, 1), 4
            )
    for col in ("key_precision", "key_recall", "key_f1"):
        metrics[f"sample_{col}"] = round(soft_df[col].mean(), 4)
    metrics["sample_n"] = len(soft_df)

    # truncation vs model effect: full-page pairs saw identical input content
    for label, subset in (
        ("full_page", both_parsed[both_parsed["full_page"]]),
        ("partial_page", both_parsed[~both_parsed["full_page"]]),
    ):
        if len(subset) == 0:
            continue
        metrics[f"{label}_pairs"] = len(subset)
        metrics[f"{label}_value_change_rate"] = round(
            subset["n_changed"].sum() / max(subset["n_shared"].sum(), 1), 4
        )

    print("\n--- D2: regenerated vs original targets ---")
    for k, v in metrics.items():
        print(f"{k:>35}: {v}")

    metrics_df = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    metrics_df.to_csv(TABLES_DIR / "regeneration_comparison.csv", index=False)
    print(f"\nSaved {TABLES_DIR / 'regeneration_comparison.csv'}")


if __name__ == "__main__":
    main()
