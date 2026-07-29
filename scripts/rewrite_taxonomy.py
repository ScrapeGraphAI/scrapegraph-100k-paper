import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import regeneration_comparison as rc  # noqa: E402

from modelling.metrics import (  # noqa: E402
    EMPTY_DICT_SENTINEL,
    _flatten_dict,
    parse_json_remove_duplicates,
)

OUT_DIR = Path(".data/rewrite_taxonomy")
TABLES_DIR = Path("tables")

LEAVES_PATH = OUT_DIR / "leaves.jsonl"
JUDGMENTS_PATH = OUT_DIR / "judgments.jsonl"
MANUAL_SHEET_PATH = OUT_DIR / "manual_sheet.csv"

MAX_VALUE_CHARS = 2_000  # cap stored/judged leaf values

# meaning-preserving surface variants the heuristics can settle without a judge
HEURISTIC_LABELS = {
    "formatting": "punctuation/diacritics/quotes/number/date formatting only",
    "reorder": "same list items in a different order",
    "completion": "regenerated value extends the original (truncation completed)",
    "truncation": "regenerated value is a strict substring of the original",
    "needs_judge": "residual — semantic comparison required",
}

DATE_HINT = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}",
    re.I,
)

JUDGE_PROMPT = """\
Two extraction systems filled the same field of a JSON extraction from the same \
web page. Decide whether the two values convey the same information.

Field path: {key}
Value A (original): {orig}
Value B (regenerated): {regen}

Verdicts:
- equivalent: same underlying fact(s); differences are only wording, \
formatting, translation, casing, abbreviation or unit expansion \
(e.g. "SEK" vs "Swedish krona"), or equally-faithful paraphrase.
- partial: the values overlap but one adds or drops material information \
(extra list items, a longer description covering more facts, a detail omitted).
- substantive: the values disagree — different entity, number, date, category \
or fact, or they describe different things entirely.
- unclear: cannot be determined from the values alone.

Return the verdict and a one-sentence reason."""

JUDGE_RESPONSE_SCHEMA = {
    "name": "rewrite_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["equivalent", "partial", "substantive", "unclear"],
            },
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    },
}


def norm_text(v) -> str:
    s = v if isinstance(v, str) else orjson.dumps(v).decode()
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\"'`´’‘“”«»()\[\]{}.,;:!?*_|/\\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def as_numbers(v) -> set[float]:
    # a lone comma is ambiguous (decimal comma vs thousands separator), so
    # return every plausible reading and match on intersection
    if isinstance(v, bool):
        return set()
    if isinstance(v, (int, float)):
        return {float(v)}
    if not isinstance(v, str):
        return set()
    stripped = v.strip()
    numeric_chars = re.sub(r"[^\d.,\-]", "", stripped)
    if not re.search(r"\d", numeric_chars):
        return set()
    if len(numeric_chars) < len(stripped) - 6:
        return set()  # mostly non-numeric text around the digits
    candidates = {numeric_chars.replace(",", "")}
    if numeric_chars.count(",") == 1 and "." not in numeric_chars:
        candidates.add(numeric_chars.replace(",", "."))
    readings = set()
    for candidate in candidates:
        with contextlib.suppress(ValueError):
            readings.add(float(candidate))
    return readings


def as_date(v):
    if not isinstance(v, str) or len(v) > 40 or not DATE_HINT.search(v):
        return None
    from dateutil import parser as date_parser

    try:
        return date_parser.parse(v, dayfirst=False, fuzzy=False).date()
    except (ValueError, OverflowError):
        return None


def heuristic_label(o, r) -> str:
    norm_o, norm_r = norm_text(o), norm_text(r)
    if norm_o and norm_o == norm_r:
        return "formatting"
    if as_numbers(o) & as_numbers(r):
        return "formatting"
    date_o, date_r = as_date(o), as_date(r)
    if date_o is not None and date_o == date_r:
        return "formatting"
    if isinstance(o, list) and isinstance(r, list) and len(o) == len(r) and o:
        canon_o = sorted(rc.canonical(x) for x in o)
        canon_r = sorted(rc.canonical(x) for x in r)
        if canon_o == canon_r:
            return "reorder"
    if norm_o and norm_r and len(norm_o) >= 15:
        if norm_o in norm_r:
            return "completion"
        if norm_r in norm_o:
            return "truncation"
    return "needs_judge"


def desentinel(v):
    # _flatten_dict marks empty dicts with a bare object() that orjson
    # cannot serialize; map it back to {} (lists of flattened dict values
    # can carry it too)
    if v is EMPTY_DICT_SENTINEL:
        return {}
    if isinstance(v, list):
        return [desentinel(x) for x in v]
    return v


def leaf_id(key: str, o, r) -> str:
    material = key.encode() + b"\x00" + rc.canonical(o) + b"\x00" + rc.canonical(r)
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def clip(v) -> str:
    s = v if isinstance(v, str) else orjson.dumps(v).decode()
    return s[:MAX_VALUE_CHARS]


def stage_collect() -> None:
    index, by_schema, raw_rows = rc.build_raw_index()
    ft = rc.load_finetuning()
    pairs = rc.align(ft, index, by_schema, raw_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    n_pairs_used = 0
    with open(LEAVES_PATH, "w") as out:
        for pair_idx, (_schema, orig_str, regen_str, full_page) in enumerate(
            tqdm(pairs, desc="Collecting rewrite leaves")
        ):
            try:
                orig = rc.normalize_root(parse_json_remove_duplicates(orig_str))
                regen = rc.normalize_root(parse_json_remove_duplicates(regen_str))
            except Exception:
                continue
            flat_orig, flat_regen = _flatten_dict(orig), _flatten_dict(regen)
            n_pairs_used += 1
            shared_keys = set(flat_orig) & set(flat_regen)
            for k in shared_keys:
                o, r = desentinel(flat_orig[k]), desentinel(flat_regen[k])
                if rc.canonical(o) == rc.canonical(r):
                    continue
                if rc.classify_change(o, r) != "chg_content_rewrite":
                    continue
                label = heuristic_label(o, r)
                counts[label] += 1
                record = {
                    "leaf_id": leaf_id(k, o, r),
                    "pair_idx": pair_idx,
                    "full_page": full_page,
                    "key": k,
                    "orig": clip(o),
                    "regen": clip(r),
                    "heuristic": label,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = sum(counts.values())
    print(f"\n{total:,} chg_content_rewrite leaves from {n_pairs_used:,} pairs")
    for label, n in counts.most_common():
        print(f"  {label:>12}: {n:,} ({n / total:.1%})")
    print(f"Saved {LEAVES_PATH}")


def load_leaves() -> list[dict]:
    if not LEAVES_PATH.exists():
        sys.exit(f"{LEAVES_PATH} not found — run the collect stage first")
    return [json.loads(line) for line in open(LEAVES_PATH)]


def load_judgments() -> list[dict]:
    if not JUDGMENTS_PATH.exists():
        return []
    records = []
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # truncated line from a killed run
                continue
    return records


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
        response_format={"type": "json_schema", "json_schema": JUDGE_RESPONSE_SCHEMA},
        **kwargs,
    )  # ty: ignore[no-matching-overload]
    raw_verdict = completion.choices[0].message.content
    return json.loads(raw_verdict)


def judge_sample(n: int, seed: int) -> list[dict]:
    residual = [lf for lf in load_leaves() if lf["heuristic"] == "needs_judge"]
    rng = random.Random(seed)
    return rng.sample(residual, min(n, len(residual)))


def stage_sheet(args) -> None:
    import csv

    sample = judge_sample(args.n_judge, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_SHEET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["leaf_id", "key", "orig", "regen", "verdict"])
        for lf in sample:
            writer.writerow([lf["leaf_id"], lf["key"], lf["orig"], lf["regen"], ""])
    print(
        f"Wrote {len(sample)} rows to {MANUAL_SHEET_PATH} — fill the verdict "
        "column with equivalent/partial/substantive/unclear; analyze picks "
        "up labeled rows (manual labels override judge verdicts)"
    )


def load_manual_labels() -> dict[str, str]:
    if not MANUAL_SHEET_PATH.exists():
        return {}
    import csv

    valid = {"equivalent", "partial", "substantive", "unclear"}
    labels = {}
    with open(MANUAL_SHEET_PATH, newline="") as f:
        for row in csv.DictReader(f):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict in valid:
                labels[row["leaf_id"]] = verdict
    return labels


def stage_judge(args) -> None:
    sample = judge_sample(args.n_judge, args.seed)

    done = {j["leaf_id"] for j in load_judgments() if "error" not in j}
    work = [lf for lf in sample if lf["leaf_id"] not in done]
    if args.limit:
        work = work[: args.limit]
    if not work:
        print("Nothing left to judge")
        return
    print(f"Judging {len(work)} of {len(sample)} sampled leaves ({len(done)} cached)")

    client = make_client()

    def run(lf: dict) -> dict:
        record = {"leaf_id": lf["leaf_id"], "judge_model": args.judge_model}
        prompt = JUDGE_PROMPT.format(key=lf["key"], orig=lf["orig"], regen=lf["regen"])
        try:  # errors surface in analyze, never kill the run
            record.update(call_judge(client, args.judge_model, prompt))
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
        return record

    with open(JUDGMENTS_PATH, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, lf) for lf in work]
        n_errors = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            record = future.result()
            if "error" in record:
                n_errors += 1
            out.write(json.dumps(record) + "\n")
            out.flush()
    print(f"Done: {len(work) - n_errors} judged, {n_errors} errors")


def stage_analyze(args) -> None:
    leaves = load_leaves()
    total = len(leaves)
    heuristic_counts = Counter(lf["heuristic"] for lf in leaves)

    labels = {j["leaf_id"]: j["verdict"] for j in load_judgments() if "error" not in j}
    manual = load_manual_labels()
    labels.update(manual)  # manual labels override judge verdicts
    residual_ids = {lf["leaf_id"] for lf in leaves if lf["heuristic"] == "needs_judge"}
    verdicts = Counter(v for lid, v in labels.items() if lid in residual_ids)
    n_judged = sum(verdicts.values())
    if manual:
        print(f"Using {len(manual)} manual labels from {MANUAL_SHEET_PATH}")

    lines = [
        "# chg_content_rewrite taxonomy",
        "",
        f"{total:,} changed shared leaves previously reported as 'genuine "
        "re-extractions or paraphrases', split by deterministic heuristics, "
        "with the residual class judged on a random sample "
        f"(n={n_judged:,}, seed={args.seed}).",
        "",
        "| class | leaves | share | meaning |",
        "|---|---|---|---|",
    ]
    for label, meaning in HEURISTIC_LABELS.items():
        n = heuristic_counts.get(label, 0)
        lines.append(f"| {label} | {n:,} | {n / total:.1%} | {meaning} |")

    if n_judged:
        residual_share = heuristic_counts.get("needs_judge", 0) / total
        verdict_order = ("equivalent", "partial", "substantive", "unclear")
        # share of ALL rewrite leaves each verdict accounts for, assuming the
        # judged sample represents the full residual class
        extrapolated = {
            verdict: residual_share * verdicts.get(verdict, 0) / n_judged
            for verdict in verdict_order
        }
        lines += [
            "",
            "Judged residual sample (share of residual / extrapolated share "
            "of all rewrite leaves):",
            "",
            "| verdict | n | of residual | of all rewrites |",
            "|---|---|---|---|",
        ]
        for verdict in verdict_order:
            n = verdicts.get(verdict, 0)
            share_of_residual = n / n_judged
            lines.append(
                f"| {verdict} | {n:,} | {share_of_residual:.1%} "
                f"| {extrapolated[verdict]:.1%} |"
            )
        n_preserving = heuristic_counts.get("formatting", 0) + heuristic_counts.get(
            "reorder", 0
        )
        preserving_share = n_preserving / total + extrapolated["equivalent"]
        completion_share = heuristic_counts.get("completion", 0) / total
        truncation_share = heuristic_counts.get("truncation", 0) / total
        partial_share = completion_share + truncation_share + extrapolated["partial"]
        substantive_share = extrapolated["substantive"]
        lines += [
            "",
            "Headline split of the former 'genuine re-extractions or "
            "paraphrases' class:",
            "",
            f"- meaning-preserving rewrites: {preserving_share:.1%}",
            f"- partial overlap (info added or dropped): {partial_share:.1%}",
            f"- substantive value changes: {substantive_share:.1%}",
            f"- unclear: {extrapolated['unclear']:.1%}",
        ]
    else:
        lines += ["", "No judgments yet — run the judge stage for the residual split."]

    TABLES_DIR.mkdir(exist_ok=True)
    out_path = TABLES_DIR / "rewrite_taxonomy.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["collect", "sheet", "judge", "analyze", "all"],
    )
    parser.add_argument("--n-judge", type=int, default=1_500)
    parser.add_argument("--judge-model", default="gpt-5.6-terra-fiit")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage in ("collect", "all"):
        stage_collect()
    if args.stage == "sheet":
        stage_sheet(args)
    if args.stage in ("judge", "all"):
        stage_judge(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)


if __name__ == "__main__":
    main()
