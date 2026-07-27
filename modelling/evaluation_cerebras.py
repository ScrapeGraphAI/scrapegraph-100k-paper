import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import wandb
from cerebras.cloud.sdk import AsyncCerebras  # ty: ignore[unresolved-import]
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from modelling import metrics as M
from modelling import prompts
from modelling.utils import extract_json

load_dotenv()

MODEL_NAME = "gpt-oss-120b"
BATCH_SIZE = 32
RESULTS_DIR = Path("sg-checkpoints/results")


@dataclass
class CerebrasEvalConfig:
    model_name: str = MODEL_NAME
    dataset_name: str = "scrapegraphai/scrapegraph-100k-finetuning"
    max_new_tokens: int = 2048 * 2
    temperature: float = 0.0
    top_p: float = 1.0
    dry_run: bool = False
    wandb_entity: str = "francesco-zuppichini"
    wandb_project: str = "scrapegraphai-100k"


async def call_cerebras(
    client: AsyncCerebras,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    config: CerebrasEvalConfig,
) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    model=config.model_name,
                    max_completion_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stream=False,
                )
                content = completion.choices[0].message.content
                if not content:
                    print(
                        f"Empty response — finish_reason={completion.choices[0].finish_reason}, usage={completion.usage}"
                    )
                    return ""
                return content
            except Exception as e:
                if attempt == 2:
                    print(f"Failed after 3 attempts: {e}")
                    return ""
                await asyncio.sleep(2**attempt)
    return ""


async def run_eval(config: CerebrasEvalConfig):
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        job_type="eval",
        tags=["eval", "cerebras"],
        config=asdict(config),
    )

    ds = load_dataset(config.dataset_name, split="test")
    assert isinstance(ds, Dataset)
    if config.dry_run:
        ds = ds.select(range(min(100, len(ds))))

    all_messages = []
    for row in ds:
        all_messages.append(
            [{"role": "user", "content": prompts.build(row["schema"], row["content"])}]
        )

    print(f"Running {len(all_messages)} samples with BATCH_SIZE={BATCH_SIZE}")

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    import os

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("Set CEREBRAS_API_KEY env var")
    client = AsyncCerebras(api_key=api_key)

    tasks = [call_cerebras(client, semaphore, msgs, config) for msgs in all_messages]
    responses = await tqdm_asyncio.gather(*tasks, desc="Cerebras inference")

    all_metrics = defaultdict(list)
    examples = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / f"{config.model_name}.jsonl"

    with open(jsonl_path, "w") as f:
        for i, (row, response) in enumerate(zip(ds, responses, strict=True)):
            schema_obj = json.loads(row["schema"])
            ground_truth = json.loads(row["response"])
            clean_response = extract_json(response) if response else ""
            sample_metrics = M.run_all(clean_response, schema_obj, ground_truth)

            for k, v in sample_metrics.items():
                all_metrics[k].append(float(v))

            record = {
                "idx": i,
                "schema": row["schema"],
                "content": row["content"],
                "ground_truth": row["response"],
                "model_response": response,
                "clean_response": clean_response,
                **{k: float(v) for k, v in sample_metrics.items()},
            }
            f.write(json.dumps(record) + "\n")

            if i < 32:
                examples.append([json.dumps(ground_truth, indent=2), clean_response])

            run.log(sample_metrics, step=i)

    print(f"Saved {len(ds)} rows to {jsonl_path}")

    examples_table = wandb.Table(columns=["ground_truth", "generated"], data=examples)
    run.log({"examples": examples_table})

    total = len(all_metrics["is_valid_json"])
    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    avg_metrics["total"] = total

    run.summary.update(avg_metrics)
    run.finish()

    print(f"\n{'='*60}")
    print(f"Results ({total} samples):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'='*60}")

    return avg_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    config = CerebrasEvalConfig(
        model_name=args.model,
        dry_run=args.dry_run,
    )
    print(f"Evaluating {config.model_name} on Cerebras")
    asyncio.run(run_eval(config))
