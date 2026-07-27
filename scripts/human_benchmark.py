import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontier_baselines import bucket_for_score, stratified_sample  # noqa: E402
from regeneration_comparison import (  # noqa: E402
    CHARS_PER_TOKEN,
    CHUNK_SIZE_TOKENS,
    MAX_CHARS,
    OVERLAP_TOKENS,
    PREFIX_PROBE_CHARS,
    chunk_in_chars,
    chunk_key,
    normalize_root,
)

from modelling import metrics as M  # noqa: E402
from modelling import prompts  # noqa: E402
from modelling.metrics import (  # noqa: E402
    _flatten_dict,
    json_validator,
    parse_json_remove_duplicates,
)

RAW_REPO = "scrapegraphai/scrapegraphai-100k"
FT_REPO = "scrapegraphai/scrapegraph-100k-finetuning"
REVISION = "v1.0"

# Local mirrors (hf_fix.py backups); fall back to the HF hub at v1.0
LOCAL_RAW = Path("hf_backup/v1/data/train.parquet")
LOCAL_FT_DIR = Path("hf_backup_scrapegraph-100k-finetuning/v1/data")

MAX_PROMPT_TOKENS = 8192
FILTER_TOKENIZER = "Qwen/Qwen3-4B"

OUT_DIR = Path(".data/human_benchmark")
TABLES_DIR = Path("tables")

INSTRUCTIONS = """\
# Human benchmark annotation protocol

Each example lives in `annotations/hb_NNN.json` with a matching
`hb_NNN.context.md` (the JSON schema followed by the page content the model
saw — for truncated chunks this is the exact truncated input, not the full
page).

For every example:

1. Read the schema and the content in the context file.
2. Open the annotation file. `gold` is pre-filled with the GPT-5-nano draft.
3. Verify every leaf value of `gold` against the content, and correct it:
   - Fix values that are wrong or paraphrased; copy values from the content.
   - Fill values that are present in the content but missing from the draft.
   - Replace hallucinated values (not supported by the content) with null.
   - If the content genuinely does not contain a requested value, use null.
   Ground truth is what is extractable from the shown content ONLY — never
   use outside knowledge, and for truncated chunks never guess at the rest
   of the page.
4. Keep `gold` schema-compliant: same structure, no keys outside the schema.
5. Set `"verified": true`. Use `"notes"` for anything ambiguous (e.g. the
   schema is a poor fit for the page, several equally defensible values).

Do not look at any model output other than the pre-filled draft.
`build` will reject unverified or schema-non-compliant annotations.
"""


def _load_table(local: Path, repo: str, filename: str, columns: list[str]):
    if local.exists():
        path = local
    else:
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=REVISION)
    table = pq.read_table(path)
    present = [c for c in columns if c in table.column_names]
    missing = sorted(set(columns) - set(present))
    if missing:
        print(f"Warning: {filename} lacks columns {missing}")
    return table.select(present)


def load_ft_split(split: str) -> list[dict]:
    table = _load_table(
        LOCAL_FT_DIR / f"{split}-00000-of-00001.parquet",
        FT_REPO,
        f"data/{split}-00000-of-00001.parquet",
        ["schema", "content", "response"],
    )
    rows = table.to_pylist()
    print(f"Finetuning {split} split: {len(rows):,} rows")
    return rows


def build_raw_index():
    """(chunk-hash -> page positions, schema -> positions, page contents, schema stats).

    Mirrors regeneration_comparison.build_raw_index but also carries the
    precomputed schema-complexity stats needed for stratification.
    """
    table = _load_table(
        LOCAL_RAW,
        RAW_REPO,
        "data/train.parquet",
        [
            "schema",
            "content",
            "response",
            "schema_complexity_score",
            "schema_depth",
            "schema_keys",
        ],
    )
    print(f"Raw dataset: {table.num_rows:,} rows")

    size = int(CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)
    overlap = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
    has_stats = "schema_complexity_score" in table.column_names

    index: dict[bytes, list[int]] = {}
    by_schema: dict[str, list[int]] = {}
    contents: list[str] = []
    schema_stats: dict[str, dict] = {}
    for batch in table.to_batches(max_chunksize=10_000):
        cols = batch.to_pydict()
        n = len(cols["schema"])
        for i in range(n):
            schema, content, response = (
                cols["schema"][i],
                cols["content"][i],
                cols["response"][i],
            )
            if has_stats and schema not in schema_stats:
                schema_stats[schema] = {
                    "complexity": cols["schema_complexity_score"][i],
                    "depth": cols["schema_depth"][i],
                    "keys": cols["schema_keys"][i],
                }
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
            pos = len(contents)
            contents.append(content)
            by_schema.setdefault(schema, []).append(pos)
            for chunk in chunk_in_chars(content, size, overlap):
                key = chunk_key(schema, chunk)
                index.setdefault(key, []).append(pos)

    print(
        f"Raw rows passing preprocess char filters: {len(contents):,}; "
        f"distinct (schema, chunk) keys: {len(index):,}"
    )
    return index, by_schema, contents, schema_stats


def align_positions(schema: str, content: str, index, by_schema, contents) -> list[int]:
    positions = index.get(chunk_key(schema, content))
    if positions:
        return positions
    probe = content[:PREFIX_PROBE_CHARS]
    return [p for p in by_schema.get(schema, []) if probe in contents[p]]


def pair_key(schema: str, response: str) -> bytes:
    return chunk_key(schema, response)  # same 16-byte blake2b keying


def length_filter(rows: list[dict], indices: list[int]) -> list[int]:
    """Reproduce the 8192-token prompt filter of modelling/evaluation.py."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FILTER_TOKENIZER)
    kept = []
    for i in indices:
        row = rows[i]
        prompt = prompts.build(row["schema"], row["content"])
        msgs = [{"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(tokens) <= MAX_PROMPT_TOKENS:
            kept.append(i)
    print(f"Length filter kept {len(kept)}/{len(indices)} candidates")
    return kept


def leaf_diff(draft: dict, gold: dict) -> dict:
    """How much the annotator changed relative to the teacher draft."""
    flat_draft = _flatten_dict(normalize_root(draft))
    flat_gold = _flatten_dict(normalize_root(gold))
    shared = set(flat_draft) & set(flat_gold)

    def canonical(v) -> bytes:
        return orjson.dumps(v, option=orjson.OPT_SORT_KEYS)

    changed = 0
    for k in shared:
        try:
            same = canonical(flat_draft[k]) == canonical(flat_gold[k])
        except TypeError:
            same = flat_draft[k] == flat_gold[k]
        changed += not same
    return {
        "n_leaves_draft": len(flat_draft),
        "n_leaves_gold": len(flat_gold),
        "n_added": len(set(flat_gold) - set(flat_draft)),
        "n_removed": len(set(flat_draft) - set(flat_gold)),
        "n_changed": changed,
        "edited": bool(changed or set(flat_gold) != set(flat_draft)),
    }


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #


def build_train_leak_state(
    train_rows: list[dict], index, by_schema, contents
) -> tuple[set[int], set[bytes], set[bytes], int]:
    """Raw pages, chunk keys, and (schema, response) pairs touched by train rows."""
    train_pages: set[int] = set()
    train_chunk_keys: set[bytes] = set()
    train_pairs: set[bytes] = set()
    n_unaligned = 0
    for row in train_rows:
        train_chunk_keys.add(chunk_key(row["schema"], row["content"]))
        train_pairs.add(pair_key(row["schema"], row["response"]))
        positions = align_positions(
            row["schema"], row["content"], index, by_schema, contents
        )
        if positions:
            train_pages.update(positions)
        else:
            n_unaligned += 1
    print(
        f"Train rows touching {len(train_pages):,} raw pages "
        f"({n_unaligned:,} train rows unalignable)"
    )
    return train_pages, train_chunk_keys, train_pairs, n_unaligned


def filter_leak_free(
    test_rows: list[dict],
    train_pages: set[int],
    train_chunk_keys: set[bytes],
    train_pairs: set[bytes],
    index,
    by_schema,
    contents,
) -> tuple[list[int], dict[int, bool], Counter]:
    """Test rows with no chunk, response, or sibling-page overlap with train."""
    drops = Counter()
    eligible: list[int] = []
    full_page_flags: dict[int, bool] = {}
    for i, row in enumerate(test_rows):
        try:
            json.loads(row["schema"])
            parse_json_remove_duplicates(row["response"])
        except Exception:
            drops["unparseable_schema_or_response"] += 1
            continue
        if chunk_key(row["schema"], row["content"]) in train_chunk_keys:
            drops["identical_chunk_in_train"] += 1
            continue
        if pair_key(row["schema"], row["response"]) in train_pairs:
            drops["identical_schema_response_in_train"] += 1
            continue
        positions = align_positions(
            row["schema"], row["content"], index, by_schema, contents
        )
        if not positions:
            drops["unalignable_provenance"] += 1
            continue
        if train_pages & set(positions):
            drops["sibling_chunk_in_train"] += 1
            continue
        eligible.append(i)
        full_page_flags[i] = all(contents[p] == row["content"] for p in positions)
    print(
        f"Leak-free test rows: {len(eligible):,}/{len(test_rows):,}; "
        f"drops: {dict(drops)}"
    )
    return eligible, full_page_flags, drops


def cmd_sample(args) -> None:
    out = Path(args.out)
    ann_dir = out / "annotations"
    sheet_path = out / "sheet.jsonl"
    if sheet_path.exists() and not args.force:
        sys.exit(
            f"{sheet_path} exists — refusing to overwrite a sheet that may "
            f"have annotations in flight (use --force to resample)"
        )

    index, by_schema, contents, schema_stats = build_raw_index()
    train_rows = load_ft_split("train")
    test_rows = load_ft_split("test")

    train_pages, train_chunk_keys, train_pairs, n_train_unaligned = (
        build_train_leak_state(train_rows, index, by_schema, contents)
    )
    eligible, full_page_flags, drops = filter_leak_free(
        test_rows,
        train_pages,
        train_chunk_keys,
        train_pairs,
        index,
        by_schema,
        contents,
    )
    n_leak_free = len(eligible)

    if not args.no_length_filter:
        eligible = length_filter(test_rows, eligible)

    strata: dict[int, str] = {}
    for i in eligible:
        stats = schema_stats.get(test_rows[i]["schema"], {})
        bucket = bucket_for_score(stats.get("complexity"))
        page_kind = "full" if full_page_flags[i] else "trunc"
        strata[i] = f"{bucket}|{page_kind}"

    sampled = stratified_sample(eligible, strata, args.n, args.seed)
    population = Counter(strata[i] for i in eligible)
    picked = Counter(strata[i] for i in sampled)
    print(f"Sampled {len(sampled)} rows over {len(picked)} strata")

    ann_dir.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w") as sheet:
        for rank, i in enumerate(sampled):
            row = test_rows[i]
            hb_id = f"hb_{rank:03d}"
            stats = schema_stats.get(row["schema"], {})
            draft = parse_json_remove_duplicates(row["response"])
            record = {
                "hb_id": hb_id,
                "test_idx": i,
                "stratum": strata[i],
                "full_page": full_page_flags[i],
                "schema_complexity": stats.get("complexity"),
                "schema_depth": stats.get("depth"),
                "schema_keys": stats.get("keys"),
                "schema": row["schema"],
                "content": row["content"],
                "draft_response": row["response"],
            }
            sheet.write(json.dumps(record) + "\n")

            annotation = {
                "hb_id": hb_id,
                "test_idx": i,
                "verified": False,
                "notes": "",
                "gold": draft,
            }
            annotation_json = json.dumps(annotation, indent=2, ensure_ascii=False)
            (ann_dir / f"{hb_id}.json").write_text(annotation_json + "\n")

            schema_obj = json.loads(row["schema"])
            schema_pretty = json.dumps(schema_obj, indent=2, ensure_ascii=False)
            context = (
                f"# {hb_id} (test_idx={i}, {strata[i]})\n\n"
                f"## Schema\n\n```json\n{schema_pretty}\n```\n\n"
                f"## Content shown to the model\n\n{row['content']}\n"
            )
            (ann_dir / f"{hb_id}.context.md").write_text(context)

    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    manifest = {
        "seed": args.seed,
        "n_requested": args.n,
        "n_sampled": len(sampled),
        "revision": REVISION,
        "test_rows": len(test_rows),
        "leak_free_rows": n_leak_free,
        "eligible_after_length_filter": len(eligible),
        "length_filter": not args.no_length_filter,
        "drops": dict(drops),
        "train_rows_unalignable": n_train_unaligned,
        "strata_population": dict(sorted(population.items())),
        "strata_sampled": dict(sorted(picked.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {sheet_path}, {len(sampled)} annotation files in {ann_dir}, "
        f"manifest and INSTRUCTIONS.md in {out}"
    )


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def load_sheet(out: Path) -> dict[str, dict]:
    with open(out / "sheet.jsonl") as f:
        records = [json.loads(line) for line in f]
    return {r["hb_id"]: r for r in records}


def cmd_build(args) -> None:
    out = Path(args.out)
    sheet = load_sheet(out)

    gold_records, diffs, problems = [], [], []
    n_unverified = 0
    for hb_id in sorted(sheet):
        path = out / "annotations" / f"{hb_id}.json"
        if not path.exists():
            problems.append(f"{hb_id}: annotation file missing")
            continue
        annotation = json.loads(path.read_text())
        if not annotation.get("verified"):
            n_unverified += 1
            continue
        row = sheet[hb_id]
        schema_obj = json.loads(row["schema"])
        gold = annotation["gold"]
        gold_json = json.dumps(gold, ensure_ascii=False)
        validation = json_validator(gold_json, schema_obj)
        if not validation["is_compliant"]:
            problems.append(f"{hb_id}: gold is not schema-compliant — fix before build")
            continue
        draft = parse_json_remove_duplicates(row["draft_response"])
        diff = leaf_diff(draft, gold)
        diffs.append(diff)
        sheet_fields = {
            k: row[k]
            for k in (
                "hb_id",
                "test_idx",
                "stratum",
                "full_page",
                "schema_complexity",
                "schema_depth",
                "schema_keys",
                "schema",
                "content",
            )
        }
        draft_diff_fields = {f"draft_{k}": v for k, v in diff.items()}
        gold_records.append(
            {
                **sheet_fields,
                "gold": gold,
                "notes": annotation.get("notes", ""),
                **draft_diff_fields,
            }
        )

    for p in problems:
        print(f"PROBLEM {p}")
    if n_unverified:
        print(f"{n_unverified} annotations still unverified — excluded")
    if not gold_records:
        sys.exit("No verified, compliant annotations — nothing to build")
    if problems and args.strict:
        sys.exit(f"{len(problems)} problems in strict mode — aborting")

    gold_path = out / "gold.jsonl"
    with open(gold_path, "w") as f:
        for r in gold_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(gold_records)
    total_shared = sum(d["n_leaves_draft"] for d in diffs)
    n_edited = sum(d["edited"] for d in diffs)
    n_changed = sum(d["n_changed"] for d in diffs)
    n_added = sum(d["n_added"] for d in diffs)
    n_removed = sum(d["n_removed"] for d in diffs)
    report = [
        "# Human benchmark build report",
        "",
        f"- verified gold examples: {n} (of {len(sheet)} sampled; "
        f"{n_unverified} unverified, {len(problems)} problems)",
        f"- examples where the annotator edited the teacher draft: "
        f"{n_edited}/{n} ({n_edited / n:.1%})",
        f"- draft leaves changed: {n_changed:,}"
        f"/{total_shared:,} ({n_changed / max(total_shared, 1):.1%})",
        f"- leaves added by annotators: {n_added:,}; " f"removed: {n_removed:,}",
        "",
        "Per-stratum verified counts:",
        "",
    ]
    by_stratum = Counter(r["stratum"] for r in gold_records)
    report += [f"- {s}: {c}" for s, c in sorted(by_stratum.items())]
    TABLES_DIR.mkdir(exist_ok=True)
    report_path = TABLES_DIR / "human_benchmark_build.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"Wrote {gold_path} ({n} rows) and {report_path}")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #

HEADLINE_METRICS = ("is_valid_json", "is_schema_compliant", "key_f1", "value_score")


def bootstrap_ci(
    values: list[float], seed: int, n_boot: int = 2000
) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot)
    )
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def cmd_score(args) -> None:
    out = Path(args.out)
    sheet = load_sheet(out)
    with open(out / "gold.jsonl") as f:
        gold_rows = [json.loads(line) for line in f]

    if args.teacher:
        name = "teacher_gpt-5-nano"
        predictions = {
            r["test_idx"]: sheet[r["hb_id"]]["draft_response"] for r in gold_rows
        }
    else:
        name = args.name or Path(args.results).stem
        predictions = {}
        with open(args.results) as f:
            for line in f:
                record = json.loads(line)
                predictions[record["idx"]] = (
                    record.get("clean_response") or record.get("model_response") or ""
                )

    records = []
    n_missing = 0
    for row in gold_rows:
        pred = predictions.get(row["test_idx"])
        if pred is None:
            n_missing += 1
            continue
        schema_obj = json.loads(row["schema"])
        sample_metrics = M.run_all(pred, schema_obj, row["gold"])
        bucket = row["stratum"].split("|")[0]
        records.append(
            {
                "stratum": row["stratum"],
                "bucket": bucket,
                "full_page": row["full_page"],
                **{k: float(v) for k, v in sample_metrics.items()},
            }
        )
    if not records:
        sys.exit("No overlap between gold rows and prediction file")
    if n_missing:
        print(
            f"Warning: {n_missing}/{len(gold_rows)} gold rows have no prediction "
            f"in {name} — scoring the covered subset only"
        )

    metric_keys = [k for k in records[0] if k not in ("stratum", "bucket", "full_page")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups["overall"].append(r)
        groups[str(r["bucket"])].append(r)
        groups["full_page" if r["full_page"] else "truncated"].append(r)

    lines = [
        f"# {name} vs human gold (n={len(records)})",
        "",
        "| group | n | " + " | ".join(metric_keys) + " |",
        "|" + "---|" * (len(metric_keys) + 2),
    ]
    for group in sorted(groups, key=lambda g: (g != "overall", g)):
        rows = groups[group]
        cells = []
        for k in metric_keys:
            group_mean = sum(r[k] for r in rows) / len(rows)
            cells.append(f"{group_mean:.4f}")
        lines.append(f"| {group} | {len(rows)} | " + " | ".join(cells) + " |")
    lines += ["", "95% bootstrap CIs (overall):", ""]
    for k in HEADLINE_METRICS:
        if k not in metric_keys:
            continue
        values = [r[k] for r in records]
        mean = sum(values) / len(values)
        lo, hi = bootstrap_ci(values, args.seed)
        lines.append(f"- {k}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    TABLES_DIR.mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    table_path = TABLES_DIR / f"human_benchmark_{safe}.md"
    table_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {table_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="draw the stratified leak-free sample")
    p_sample.add_argument("--n", type=int, default=100)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--out", default=str(OUT_DIR))
    p_sample.add_argument("--no-length-filter", action="store_true")
    p_sample.add_argument("--force", action="store_true")
    p_sample.set_defaults(func=cmd_sample)

    p_build = sub.add_parser(
        "build", help="collect verified annotations into gold.jsonl"
    )
    p_build.add_argument("--out", default=str(OUT_DIR))
    p_build.add_argument("--strict", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_score = sub.add_parser("score", help="score a model's outputs against human gold")
    p_score.add_argument("--out", default=str(OUT_DIR))
    p_score.add_argument("--results", help="per-sample results JSONL keyed by test idx")
    p_score.add_argument(
        "--teacher",
        action="store_true",
        help="score the GPT-5-nano draft targets themselves",
    )
    p_score.add_argument("--name", help="model name for the output table")
    p_score.add_argument("--seed", type=int, default=42)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    if args.cmd == "score" and not args.teacher and not args.results:
        parser.error("score requires --results or --teacher")
    args.func(args)


if __name__ == "__main__":
    main()
