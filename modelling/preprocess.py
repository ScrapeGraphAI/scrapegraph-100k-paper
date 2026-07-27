import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset
from dotenv import load_dotenv

from modelling.consts import DATA_DIR

INPUT_PATH = "/media/zuppif/Cancro/scrapegraph-dataset/data/data_balanced.parquet"
SEED = 42
TEST_RATIO = 0.1
CHARS_PER_TOKEN = 3.5
CHUNK_SIZE_TOKENS = 4096
OVERLAP_TOKENS = 128
MAX_CHARS = {"content": 50_000, "schema": 10_000, "response": 10_000}


def _char_filter(table, column: str, max_chars: int):
    values = table[column]
    not_null = pc.is_valid(values)  # ty: ignore[unresolved-attribute]
    null_safe = pc.if_else(not_null, values, "")  # ty: ignore[unresolved-attribute]
    lengths = pc.utf8_length(null_safe)  # ty: ignore[unresolved-attribute]
    within = pc.less_equal(lengths, max_chars)  # ty: ignore[unresolved-attribute]
    mask = pc.and_(not_null, within)  # ty: ignore[unresolved-attribute]
    return table.filter(mask)


def chunk_in_chars(src: str, size: int, overlap: int) -> list[str]:
    if not src:
        return []
    step = size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")
    chunks = []
    for i in range(0, len(src), step):
        chunks.append(src[i : i + size])
    return chunks


def chunk_in_tokens(src: str, size: int, overlap: int) -> list[str]:
    size_in_chars = int(size * CHARS_PER_TOKEN)
    overlap_in_chars = int(overlap * CHARS_PER_TOKEN)
    return chunk_in_chars(src, size_in_chars, overlap_in_chars)


def chunk_dataset(ds: Dataset, chunk_size: int, overlap: int) -> Dataset:
    rows = []
    for row in ds:
        content_chunks = chunk_in_tokens(row["content"], chunk_size, overlap)
        for chunk in content_chunks:
            rows.append(
                {
                    "schema": row["schema"],
                    "content": chunk,
                    "response": row["response"],
                }
            )
    return Dataset.from_list(rows)


def preprocessing() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    chunk_size = CHUNK_SIZE_TOKENS
    overlap = OVERLAP_TOKENS

    table = pq.read_table(INPUT_PATH)
    print(f"Original: {table.num_rows} rows")

    for col in ["content", "schema", "response"]:
        before = table.num_rows
        table = _char_filter(table, col, MAX_CHARS[col])
        print(
            f"  {col}: removed {before - table.num_rows} rows (max {MAX_CHARS[col]:,} chars)"
        )

    table = table.select(["schema", "content", "response"])
    print(f"Filtered: {table.num_rows} rows")

    ds = Dataset(table)

    print(f"Chunking content (size={chunk_size} tokens, overlap={overlap} tokens)...")
    ds = chunk_dataset(ds, chunk_size, overlap)
    print(f"After chunking: {len(ds)} rows")

    splits = ds.train_test_split(test_size=TEST_RATIO, seed=SEED, shuffle=True)
    print(f"Train: {len(splits['train'])} | Test: {len(splits['test'])}")

    splits["train"].to_parquet(DATA_DIR / "train.parquet", compression="zstd")
    splits["test"].to_parquet(DATA_DIR / "test.parquet", compression="zstd")
    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    load_dotenv()
    preprocessing()
