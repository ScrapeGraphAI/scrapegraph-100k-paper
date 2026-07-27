import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .consts import DATA_DIR
from .metrics import fulfills_schema
from .prompts import build
from .utils import extract_json

MODEL = "gpt-5-nano-2025-08-07"
CHUNK_SIZE = 10000

CUSTOM_ID_PREFIX = "sgai-100k"
CUSTOM_ID_SEP = "::"


def jsonl_path(split: str, chunk: int) -> Path:
    return DATA_DIR / f"batch_input_{split}_{chunk}.jsonl"


def state_path(split: str) -> Path:
    return DATA_DIR / f"batch_state_{split}.json"


def output_jsonl_path(split: str, chunk: int) -> Path:
    return DATA_DIR / f"batch_output_{split}_{chunk}.jsonl"


def final_parquet_path(split: str) -> Path:
    return DATA_DIR / f"{split}_regenerated.parquet"


def input_parquet_path(split: str) -> Path:
    return DATA_DIR / f"{split}.parquet"


def load_state(split: str) -> dict | None:
    p = state_path(split)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_state(split: str, state: dict) -> None:
    state_path(split).write_text(json.dumps(state, indent=2))


def parse_idx(custom_id: str) -> int:
    return int(custom_id.rsplit(CUSTOM_ID_SEP, 1)[1])


def generate_jsonl(split: str) -> list[Path]:
    df = pd.read_parquet(input_parquet_path(split))
    n_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    paths = []

    for chunk in range(n_chunks):
        start = chunk * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(df))
        out = jsonl_path(split, chunk)

        with open(out, "w") as f:
            for idx in tqdm(
                range(start, end), desc=f"Generating {split} chunk {chunk}"
            ):
                row = df.iloc[idx]
                prompt = build(row["schema"], row["content"])
                request = {
                    "custom_id": f"{CUSTOM_ID_PREFIX}{CUSTOM_ID_SEP}{split}{CUSTOM_ID_SEP}{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                f.write(json.dumps(request) + "\n")

        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"Generated {out} ({size_mb:.2f} MB, {end - start} requests)")
        if size_mb > 200:
            raise ValueError(
                f"Chunk {chunk} too large: {size_mb:.2f} MB > 200 MB limit"
            )
        paths.append(out)

    return paths


def upload_and_create_batches(client: OpenAI, split: str, paths: list[Path]) -> dict:
    batches = []

    for chunk, jsonl in enumerate(paths):
        print(f"Uploading {jsonl}...")
        with open(jsonl, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"Uploaded: {file_obj.id}")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"split": split, "chunk": str(chunk)},
        )
        print(f"Batch {chunk} created: {batch.id} (status: {batch.status})")

        batches.append(
            {
                "chunk": chunk,
                "batch_id": batch.id,
                "input_file_id": file_obj.id,
                "status": batch.status,
            }
        )

    state = {"batches": batches}
    save_state(split, state)
    return state


def check_status(client: OpenAI, split: str) -> dict:
    state = load_state(split)
    if not state:
        raise ValueError(f"No state for {split}")

    total, completed, failed = 0, 0, 0
    all_done = True

    for b in state["batches"]:
        batch = client.batches.retrieve(b["batch_id"])
        b["status"] = batch.status
        b["output_file_id"] = batch.output_file_id
        b["error_file_id"] = batch.error_file_id

        counts = batch.request_counts
        assert counts is not None
        total += counts.total
        completed += counts.completed
        failed += counts.failed

        if batch.status not in ("completed", "failed", "expired", "cancelled"):
            all_done = False

        print(
            f"[{split}] Chunk {b['chunk']}: {batch.status} ({counts.completed}/{counts.total})"
        )

    save_state(split, state)
    print(f"[{split}] Total: {completed}/{total} ({failed} failed)")

    state["all_done"] = all_done
    state["all_completed"] = all(b["status"] == "completed" for b in state["batches"])
    return state


def download_outputs(client: OpenAI, split: str) -> list[Path]:
    state = load_state(split)
    if not state:
        raise ValueError(f"No state for {split}")

    paths = []
    for b in state["batches"]:
        if not b.get("output_file_id"):
            print(f"Chunk {b['chunk']} has no output (status: {b['status']})")
            continue

        out = output_jsonl_path(split, b["chunk"])
        print(f"Downloading chunk {b['chunk']} to {out}...")

        content = client.files.content(b["output_file_id"])
        out.write_bytes(content.read())
        print(f"Downloaded {out.stat().st_size / (1024*1024):.2f} MB")
        paths.append(out)

    return paths


def build_dataset(split: str) -> Path:
    df = pd.read_parquet(input_parquet_path(split))
    state = load_state(split)
    if not state:
        raise ValueError(f"No state for {split}")

    n_rows = len(df)
    success, errors = 0, 0

    for b in state["batches"]:
        output_jsonl = output_jsonl_path(split, b["chunk"])
        if not output_jsonl.exists():
            print(f"Skipping chunk {b['chunk']}: output not found")
            continue

        with open(output_jsonl) as f:
            for line in tqdm(f, desc=f"Parsing {split} chunk {b['chunk']}"):
                try:
                    entry = json.loads(line)
                    idx = parse_idx(entry["custom_id"])

                    if entry.get("error") or entry["response"]["status_code"] != 200:
                        errors += 1
                        continue

                    if idx >= n_rows:
                        errors += 1
                        continue

                    message_content = entry["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                    parsed = extract_json(message_content)
                    df.at[idx, "response"] = json.dumps(parsed)
                    success += 1
                except Exception:
                    errors += 1

    print(f"Parsed {success} results, {errors} errors")

    original_count = len(df)
    valid_mask = []
    for idx in tqdm(range(len(df)), desc="Validating against schema"):
        row = df.iloc[idx]
        try:
            schema = (
                json.loads(row["schema"])
                if isinstance(row["schema"], str)
                else row["schema"]
            )
            response = (
                json.loads(row["response"])
                if isinstance(row["response"], str)
                else row["response"]
            )
            is_valid, _ = fulfills_schema(schema, response)
            valid_mask.append(is_valid)
        except Exception:
            valid_mask.append(False)

    df = df[valid_mask].reset_index(drop=True)
    invalid_count = original_count - len(df)
    invalid_pct = 100 * invalid_count / original_count
    print(
        f"Filtered {invalid_count}/{original_count} invalid responses ({invalid_pct:.2f}%)"
    )

    out = final_parquet_path(split)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"Saved {out} ({len(df)} rows)")
    return out


def find_jsonl_chunks(split: str) -> list[Path]:
    return sorted(DATA_DIR.glob(f"batch_input_{split}_*.jsonl"))


def all_outputs_exist(split: str) -> bool:
    state = load_state(split)
    if not state:
        return False
    return all(output_jsonl_path(split, b["chunk"]).exists() for b in state["batches"])


def process_split(client: OpenAI, split: str) -> None:
    print(f"\n{'='*40}")
    print(f"Processing: {split}")
    print(f"{'='*40}")

    if not input_parquet_path(split).exists():
        print(f"Input not found: {input_parquet_path(split)}")
        return

    if final_parquet_path(split).exists():
        print(f"Already done: {final_parquet_path(split)}")
        return

    if all_outputs_exist(split):
        print("All outputs exist, building dataset...")
        build_dataset(split)
        return

    state = load_state(split)

    if state:
        state = check_status(client, split)
        if state["all_completed"]:
            download_outputs(client, split)
            build_dataset(split)
        elif state["all_done"]:
            print("Some batches failed/expired. Check state file.")
        else:
            print("Batches still processing. Run again later.")
        return

    chunks = find_jsonl_chunks(split)
    if chunks:
        print(f"Found {len(chunks)} JSONL chunks, uploading...")
        upload_and_create_batches(client, split, chunks)
        return

    print("Starting fresh: generating JSONL chunks...")
    paths = generate_jsonl(split)
    upload_and_create_batches(client, split, paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate responses using OpenAI Batch API"
    )
    parser.add_argument(
        "--split", choices=["train", "test"], help="Process single split"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check status, don't start new batches",
    )
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    splits = [args.split] if args.split else ["train", "test"]

    for split in splits:
        if args.status_only:
            state = load_state(split)
            if state:
                check_status(client, split)
            else:
                print(f"[{split}] No batch in progress")
        else:
            process_split(client, split)


if __name__ == "__main__":
    main()
