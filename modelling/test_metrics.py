import orjson
import pandas as pd
from datasets import Dataset, load_dataset

from modelling.consts import HF_DATASET_FINETUNING_NAME
from modelling.metrics import (
    json_validator,
    magic_metric,
    magic_metric_failed,
    parse_json_remove_duplicates,
)


def main() -> dict:
    ds = load_dataset(HF_DATASET_FINETUNING_NAME, split="test")
    assert isinstance(ds, Dataset)

    scores = []

    for row in ds:
        response = row["response"]  # this should be output from model
        schema = orjson.loads(row["schema"])
        true = orjson.loads(row["response"])
        validation = json_validator(response, schema)

        if not validation["is_valid"] or not validation["is_compliant"]:
            result = magic_metric_failed()
        else:
            obj = parse_json_remove_duplicates(response)
            result = magic_metric(obj, true)
        scores.append({**validation, **result})

    df = pd.DataFrame(scores)
    validity_rates = df[["is_valid", "is_compliant"]].mean().to_dict()
    passing_rows = df[(df["is_valid"]) & (df["is_compliant"])]
    metric_columns = passing_rows.drop(columns=["is_valid", "is_compliant", "error"])
    avg_metrics = metric_columns.mean().to_dict()
    summary = {**validity_rates, **avg_metrics}
    print(summary)
    return summary


if __name__ == "__main__":
    main()
