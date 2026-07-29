import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontier_baselines import (  # noqa: E402
    BUCKET_LABELS,
    DEPTH_GROUP,
    DEPTH_THRESHOLD,
    KEYS_GROUP,
    KEYS_THRESHOLD,
    UNKNOWN_BUCKET,
)

TABLES_DIR = Path("tables")

GROUP_ORDER = ["overall", *BUCKET_LABELS, UNKNOWN_BUCKET, DEPTH_GROUP, KEYS_GROUP]


def load_records(path: Path) -> dict[int, dict]:
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records[r["idx"]] = r
    return records


def groups_for(r: dict) -> list[str]:
    groups = ["overall", str(r["complexity_bucket"])]
    if (r.get("schema_depth") or 0) >= DEPTH_THRESHOLD:
        groups.append(DEPTH_GROUP)
    if (r.get("schema_keys") or 0) >= KEYS_THRESHOLD:
        groups.append(KEYS_GROUP)
    return groups


def normalize_error(error: str) -> str:
    msg = error.removeprefix("schema_rejected: ")
    # proxy errors are multiply-escaped JSON; strip the escaping, then pull
    # Anthropic's actual cause ('output_format.schema: <reason>' or the bare
    # invalid_request_error message) so per-request noise collapses
    msg = msg.replace("\\\\", "\\").replace("\\'", "'").replace('\\"', '"')
    m = re.search(r'output_format\.schema: ([^"]+)', msg) or re.search(
        r'"invalid_request_error","message":"([^"]+)"', msg
    )
    if m:
        msg = m.group(1)
    else:
        msg = re.sub(r"req(uest)?[_ ]?id[\"':=\s]+\S+", "", msg, flags=re.I)
        msg = re.sub(r"['\"][^'\"]{40,}['\"]", "'<snip>'", msg)
    msg = re.sub(r"\d{4,}", "<n>", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg[:220]


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def is_rate_limit(error: str) -> bool:
    # provider-side grammar-compilation throttling (e.g. Anthropic's 20/min):
    # a transient harness artifact, not a schema-capability rejection
    low = error.lower()
    return "rate limit" in low or "try again later" in low


def build_table(records: dict[int, dict], planned: dict[int, dict] | None) -> list[str]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records.values():
        for g in groups_for(r):
            by_group[g].append(r)
    planned_by_group: dict[str, int] = {}
    if planned:
        for r in planned.values():
            for g in groups_for(r):
                planned_by_group[g] = planned_by_group.get(g, 0) + 1

    cols = [
        "n_planned",
        "n_done",
        "n_rej_ratelimit",
        "n_rej_schema",
        "n_accepted",
        "compliant_all",
        "keyF1_all",
        "compliant_accepted",
        "keyF1_accepted",
        "e2e_strict_done",
        "e2e_strict_planned",
    ]
    lines = ["| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for group in GROUP_ORDER:
        rows = by_group.get(group)
        if not rows:
            continue
        rejected = [r for r in rows if r.get("schema_rejected")]
        accepted = [r for r in rows if not r.get("schema_rejected")]
        rej_rl = [r for r in rejected if is_rate_limit(r.get("api_error", ""))]
        n_planned = planned_by_group.get(group, len(rows))
        # strict end-to-end: provider-side schema rejection counts as failure
        # even though the fallback json_object call produced (and scored) output
        accepted_compliant = sum(r["is_schema_compliant"] for r in accepted)
        cells = [
            str(n_planned) if planned else "-",
            str(len(rows)),
            str(len(rej_rl)),
            str(len(rejected) - len(rej_rl)),
            str(len(accepted)),
            f"{mean(rows, 'is_schema_compliant'):.4f}",
            f"{mean(rows, 'key_f1'):.4f}",
            f"{mean(accepted, 'is_schema_compliant'):.4f}",
            f"{mean(accepted, 'key_f1'):.4f}",
            f"{accepted_compliant / len(rows):.4f}",
            f"{accepted_compliant / n_planned:.4f}" if planned else "-",
        ]
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    return lines


def rejection_causes(records: dict[int, dict], top: int) -> list[str]:
    rejected = [r for r in records.values() if r.get("schema_rejected")]
    if not rejected:
        return ["No schema rejections recorded."]
    causes = Counter(normalize_error(r.get("api_error", "")) for r in rejected)
    lines = [
        f"{len(rejected)} rejected calls, {len(causes)} distinct causes "
        f"(normalized). Top {min(top, len(causes))}:",
        "",
        "| n | cause |",
        "|---|---|",
    ]
    for cause, n in causes.most_common(top):
        lines.append(f"| {n} | {cause or '<empty error message>'} |")

    accepted = [r for r in records.values() if not r.get("schema_rejected")]
    lines += [
        "",
        "| population | n | mean schema_keys | mean schema_depth |",
        "|---|---|---|---|",
    ]
    for label, rows in (("rejected", rejected), ("accepted", accepted)):
        keys = [r["schema_keys"] for r in rows if r.get("schema_keys") is not None]
        depth = [r["schema_depth"] for r in rows if r.get("schema_depth") is not None]
        lines.append(
            f"| {label} | {len(rows)} "
            f"| {sum(keys) / len(keys):.1f} | {sum(depth) / len(depth):.2f} |"
        )
    return lines


def subset_sensitivity(
    records: dict[int, dict], planned: dict[int, dict]
) -> list[str]:
    # is the covered subset representative? score the reference model on its
    # full planned set vs restricted to the indices this run completed
    covered = {idx: r for idx, r in planned.items() if idx in records}
    ref_model = planned[next(iter(planned))].get("model", "reference")
    by_group_full: dict[str, list[dict]] = defaultdict(list)
    by_group_covered: dict[str, list[dict]] = defaultdict(list)
    for idx, r in planned.items():
        for g in groups_for(r):
            by_group_full[g].append(r)
            if idx in covered:
                by_group_covered[g].append(r)

    lines = [
        f"{ref_model} on its full set vs restricted to the "
        f"{len(covered)}/{len(planned)} indices this run completed "
        f"({len(covered) / len(planned):.1%} coverage):",
        "",
        "| group | coverage | compliant full | compliant covered | delta "
        "| keyF1 full | keyF1 covered | delta |",
        "|" + "---|" * 8,
    ]
    for group in GROUP_ORDER:
        full = by_group_full.get(group)
        sub = by_group_covered.get(group)
        if not full or not sub:
            continue
        c_full, c_sub = mean(full, "is_schema_compliant"), mean(
            sub, "is_schema_compliant"
        )
        f_full, f_sub = mean(full, "key_f1"), mean(sub, "key_f1")
        lines.append(
            f"| {group} | {len(sub)}/{len(full)} "
            f"| {c_full:.4f} | {c_sub:.4f} | {c_sub - c_full:+.4f} "
            f"| {f_full:.4f} | {f_sub:.4f} | {f_sub - f_full:+.4f} |"
        )
    return lines


def config_block(results_path: Path) -> list[str]:
    summary_path = results_path.with_name(results_path.stem + "_summary.json")
    if not summary_path.exists():
        return [f"No summary file found at {summary_path} — config not documented."]
    config = json.loads(summary_path.read_text()).get("config", {})
    return ["```json", json.dumps(config, indent=2), "```"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--top-causes", type=int, default=10)
    args = parser.parse_args()

    results_path = Path(args.results)
    records = load_records(results_path)
    planned = None
    if args.reference:
        planned = load_records(Path(args.reference))
        missing = sorted(set(planned) - set(records))
        extra = sorted(set(records) - set(planned))
        print(
            f"Coverage vs reference: {len(records)}/{len(planned)} "
            f"({len(missing)} missing, {len(extra)} not in reference)"
        )

    model = records[next(iter(records))].get("model", results_path.stem)
    lines = [
        f"# Schema-rejection accounting: {model}",
        "",
        "Denominators: 'all' metrics average over every completed call "
        "(rejected schemas fall back to json_object, so they still produce "
        "scored output); 'accepted' averages only over calls whose schema the "
        "provider accepted in json_schema mode; 'e2e_strict' treats provider-"
        "side schema rejection as failure (metric = 0). 'e2e_strict_planned' "
        "additionally treats never-completed calls as failure.",
        "",
        *build_table(records, planned),
        "",
    ]
    if planned:
        lines += ["## Coverage sensitivity", "", *subset_sensitivity(records, planned), ""]
    lines += [
        "## Rejection causes",
        "",
        *rejection_causes(records, args.top_causes),
        "",
        "## Run configuration",
        "",
        *config_block(results_path),
    ]

    TABLES_DIR.mkdir(exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    out_path = TABLES_DIR / f"rejection_report_{safe_model}.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
