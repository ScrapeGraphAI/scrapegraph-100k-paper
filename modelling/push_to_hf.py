import argparse

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

from modelling.consts import DATA_DIR, HF_DATASET_FINETUNING_NAME


def push(use_regenerated: bool = False) -> None:
    suffix = "_regenerated" if use_regenerated else ""
    train_path = DATA_DIR / f"train{suffix}.parquet"
    test_path = DATA_DIR / f"test{suffix}.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing {train_path} or {test_path}")

    print(f"Loading {train_path} and {test_path}...")
    splits = DatasetDict(
        {
            "train": Dataset.from_parquet(str(train_path)),
            "test": Dataset.from_parquet(str(test_path)),
        }
    )
    print(f"Train: {len(splits['train'])} | Test: {len(splits['test'])}")

    print(f"Pushing to {HF_DATASET_FINETUNING_NAME}...")
    splits.push_to_hub(HF_DATASET_FINETUNING_NAME, private=False)
    print(f"Pushed to {HF_DATASET_FINETUNING_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument(
        "--regenerated", action="store_true", help="Use regenerated parquet files"
    )
    args = parser.parse_args()

    load_dotenv()
    push(args.regenerated)
