import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATASET_ID = "scrapegraphai/scrapegraphai-100k"
REVISION = "v1.0"
EXPECTED_ROWS = 93695
FIELDS = ("prompt", "content", "schema", "response")

PII_LABELS = (
    "private_email",
    "private_phone",
    "private_address",
    "private_person",
    "private_url",
    "private_date",
    "account_number",
    "secret",
)

# High-entropy credential formats that token classifiers can miss. Each pattern is
# specific enough to be near-100% precision on web text.
SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "stripe_key": re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "jwt": re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

# Keep chunks well under the model's 128k-token context (~4 chars/token).
CHUNK_CHARS = 100_000
CHUNK_OVERLAP = 1_000
SNIPPET_LEN = 80


def parse_args():
    p = argparse.ArgumentParser(description="PII audit of the raw dataset")
    p.add_argument("--out-dir", type=Path, default=Path("pii_audit_out"))
    p.add_argument("--device", default="cuda", help="cuda | cpu (default: cuda)")
    p.add_argument("--limit", type=int, default=None, help="only scan the first N rows")
    p.add_argument(
        "--resume", action="store_true", help="continue from the last checkpoint"
    )
    return p.parse_args()


def load_data():
    from datasets import Dataset, load_dataset

    ds = load_dataset(DATASET_ID, revision=REVISION, split="train")
    assert isinstance(ds, Dataset)
    if len(ds) != EXPECTED_ROWS:
        sys.exit(
            f"ERROR: expected {EXPECTED_ROWS} rows at revision {REVISION}, got {len(ds)}. "
            "Refusing to audit the wrong dataset version."
        )
    return ds


def chunk_text(text):
    """Yield (offset, chunk) pairs covering text, with overlap so boundary spans survive."""
    if len(text) <= CHUNK_CHARS:
        yield 0, text
        return
    start = 0
    while start < len(text):
        yield start, text[start : start + CHUNK_CHARS]
        start += CHUNK_CHARS - CHUNK_OVERLAP


def model_spans(pf, text):
    for offset, chunk in chunk_text(text):
        result = pf.redact(chunk)
        for span in result.detected_spans:
            yield {
                "label": span.label,
                "start": offset + span.start,
                "end": offset + span.end,
                "snippet": span.text[:SNIPPET_LEN],
                "detector": "privacy-filter",
            }


def regex_spans(text):
    for name, pattern in SECRET_PATTERNS.items():
        for m in pattern.finditer(text):
            yield {
                "label": "secret",
                "start": m.start(),
                "end": m.end(),
                "snippet": m.group()[:SNIPPET_LEN],
                "detector": f"regex:{name}",
            }


def scan_field(pf, text):
    """All spans in one field, deduped: regex hits inside a model span are dropped."""
    spans = list(model_spans(pf, text))
    covered = [(s["start"], s["end"]) for s in spans]
    for rs in regex_spans(text):
        inside_model_span = any(a <= rs["start"] and rs["end"] <= b for a, b in covered)
        if not inside_model_span:
            spans.append(rs)
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans


def audit(args, ds, findings_path, checkpoint_path):
    from opf import OPF  # ty: ignore[unresolved-import]
    from tqdm import tqdm

    pf = OPF(device=args.device)

    start_row = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        start_row = checkpoint["rows_done"]
        print(f"Resuming from row index {start_row}")
    elif findings_path.exists() and not args.resume:
        sys.exit(
            f"{findings_path} exists - pass --resume to continue or delete the directory."
        )

    total = min(args.limit, len(ds)) if args.limit else len(ds)
    n_findings = 0
    # ~49% of cells are exact duplicates of another cell (schemas especially), so
    # cache spans by text hash and never scan the same text twice.
    cache = {}
    cache_hits = 0
    with open(findings_path, "a", encoding="utf-8") as out:
        progress = tqdm(
            range(start_row, total), initial=start_row, total=total, unit="row"
        )
        for i in progress:
            row = ds[i]
            for field in FIELDS:
                text = row[field]
                if not text:
                    continue
                text = str(text)
                key = hashlib.sha1(text.encode("utf-8")).digest()
                if key in cache:
                    spans = cache[key]
                    cache_hits += 1
                else:
                    spans = scan_field(pf, text)
                    cache[key] = spans
                for span in spans:
                    record = {
                        "row_id": row["id"],
                        "row_index": i,
                        "field": field,
                        **span,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_findings += 1
            out.flush()
            checkpoint_path.write_text(json.dumps({"rows_done": i + 1}))
            progress.set_postfix(findings=n_findings, cache_hits=cache_hits)
    return total


def load_findings(findings_path):
    """Read findings back, deduping anything double-written across a resume boundary."""
    seen, findings = set(), []
    with open(findings_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["row_id"], rec["field"], rec["label"], rec["start"], rec["end"])
            if key not in seen:
                seen.add(key)
                findings.append(rec)
    return findings


def write_tables(findings, rows_scanned, out_dir):
    span_counts = defaultdict(int)
    row_sets = defaultdict(set)
    for rec in findings:
        span_counts[(rec["label"], rec["field"])] += 1
        row_sets[(rec["label"], rec["field"])].add(rec["row_id"])

    def table(fmt_cell):
        header = "| PII class | " + " | ".join(FIELDS) + " | total |"
        separator = "|---" * (len(FIELDS) + 2) + "|"
        lines = [header, separator]
        for label in PII_LABELS:
            cells = [fmt_cell(label, f) for f in FIELDS]
            cells_md = " | ".join(str(c) for c in cells)
            lines.append(f"| {label} | {cells_md} | {sum(cells)} |")
        return "\n".join(lines)

    md = [
        f"# PII audit - {DATASET_ID} @ {REVISION}",
        "",
        f"Rows scanned: {rows_scanned} / {EXPECTED_ROWS}. Fields: {', '.join(FIELDS)}.",
        f"Detectors: openai/privacy-filter + {len(SECRET_PATTERNS)} secret regexes.",
        "",
        "## Detected spans (total occurrences)",
        "",
        table(lambda label, field: span_counts[(label, field)]),
        "",
        "## Affected rows (distinct rows with >=1 hit)",
        "",
        table(lambda label, field: len(row_sets[(label, field)])),
        "",
    ]
    (out_dir / "audit_table.md").write_text("\n".join(md), encoding="utf-8")

    with open(out_dir / "audit_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pii_class", "field", "span_count", "affected_rows"])
        for label in PII_LABELS:
            for field in FIELDS:
                w.writerow(
                    [
                        label,
                        field,
                        span_counts[(label, field)],
                        len(row_sets[(label, field)]),
                    ]
                )

    print("\n".join(md))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    findings_path = args.out_dir / "findings.jsonl"
    checkpoint_path = args.out_dir / "checkpoint.json"

    print(f"Loading {DATASET_ID} @ {REVISION} ...")
    ds = load_data()
    print(f"OK: {len(ds)} rows.")

    rows_scanned = audit(args, ds, findings_path, checkpoint_path)
    findings = load_findings(findings_path)
    write_tables(findings, rows_scanned, args.out_dir)


if __name__ == "__main__":
    main()
