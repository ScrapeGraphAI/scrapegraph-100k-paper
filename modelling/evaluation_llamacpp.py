import asyncio
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import httpx

BIN_DIR = "/app"
HF_DATASET_FINETUNING_NAME = "scrapegraphai/scrapegraph-100k-finetuning"
SERVER_PORT = 8080
MAX_CONCURRENT = 8


@dataclass
class EvalConfig:
    model_path: str = ""
    dataset_name: str = HF_DATASET_FINETUNING_NAME
    max_tokens: int = 4096
    context_size: int = 8192 + 4096
    temperature: float = 0.0
    repetition_penalty: float = 1.1
    num_samples: int | None = None
    dry_run: bool = False
    wandb_entity: str = "francesco-zuppichini"
    wandb_project: str = "scrapegraphai-100k"


app = modal.App("scrapegraph-eval-llamacpp")

eval_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:full-cuda", add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []", "CMD []")
    .uv_pip_install(
        "datasets",
        "hf-transfer",
        "httpx",
        "huggingface_hub",
        "jsonschema_rs",
        "orjson",
        "sacrebleu",
        "tqdm",
        "wandb",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("modelling")
)

model_cache_volume = modal.Volume.from_name("sg-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("sg-checkpoints", create_if_missing=True)

GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 4


def start_server(model_path: str, ctx: int) -> subprocess.Popen:
    cmd = [
        f"{BIN_DIR}/llama-server",
        "-m",
        model_path,
        "--port",
        str(SERVER_PORT),
        "-ngl",
        "-1",
        "-c",
        str(ctx * MAX_CONCURRENT),
        "-np",
        str(MAX_CONCURRENT),
    ]
    print(f"Starting llama-server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def wait_for_server(timeout: int = 120) -> bool:
    import httpx

    url = f"http://localhost:{SERVER_PORT}/v1/models"
    for _ in range(timeout):
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1)
    return False


async def infer_one(
    idx: int,
    row: dict,
    config: EvalConfig,
    client: "httpx.AsyncClient",
    sem: asyncio.Semaphore,
) -> dict:
    from modelling import prompts

    async with sem:
        try:
            resp = await client.post(
                f"http://localhost:{SERVER_PORT}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [
                        {"role": "system", "content": "/no_think"},
                        {
                            "role": "user",
                            "content": prompts.build(row["schema"], row["content"]),
                        },
                    ],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "repeat_penalty": config.repetition_penalty,
                },
            )
            body = resp.json()
            if "choices" not in body:
                return {"idx": idx, "error": json.dumps(body)[:300]}
            return {"idx": idx, "response": body["choices"][0]["message"]["content"]}
        except Exception as e:
            return {"idx": idx, "error": str(e)[:300]}


def collect_metrics(results: list[dict], ds, jsonl_path: str):
    from modelling import metrics as M
    from modelling.utils import extract_json

    all_metrics = defaultdict(list)
    examples = []
    errors = 0

    for result in results:
        idx = result["idx"]
        row = ds[idx]

        if "error" in result:
            print(f"  [{idx}] error: {result['error']}")
            errors += 1
            continue

        response = result["response"]
        clean_response = extract_json(response)
        schema_obj = json.loads(row["schema"])
        ground_truth = json.loads(row["response"])
        sample_metrics = M.run_all(clean_response, schema_obj, ground_truth)

        for k, v in sample_metrics.items():
            all_metrics[k].append(float(v))

        record = {
            "idx": idx,
            "schema": row["schema"],
            "content": row["content"],
            "ground_truth": row["response"],
            "model_response": response,
            "clean_response": clean_response,
            **{k: float(v) for k, v in sample_metrics.items()},
        }
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if len(examples) < 32:
            examples.append([json.dumps(ground_truth, indent=2), clean_response])

    return all_metrics, examples, errors


async def run_eval_async(config: EvalConfig):
    import httpx
    import wandb
    from datasets import Dataset, load_dataset

    server = start_server(config.model_path, config.context_size)

    if not wait_for_server():
        stderr = server.stderr.read().decode() if server.stderr else ""
        server.kill()
        raise RuntimeError(f"llama-server failed to start.\nstderr: {stderr[:2000]}")

    try:
        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            job_type="eval-gguf",
            tags=["eval", "gguf", Path(config.model_path).stem],
            config=asdict(config),
        )

        ds = load_dataset(config.dataset_name, split="test")
        assert isinstance(ds, Dataset)

        if config.dry_run:
            ds = ds.select(range(min(100, len(ds))))
        elif config.num_samples:
            ds = ds.shuffle(seed=42).select(range(min(config.num_samples, len(ds))))

        run_identifier = run.name or run.id
        results_dir = "/checkpoints/results"
        os.makedirs(results_dir, exist_ok=True)
        jsonl_path = f"{results_dir}/{run_identifier}.jsonl"
        open(jsonl_path, "w").close()

        print(f"Sending {len(ds)} requests with {MAX_CONCURRENT} concurrent slots")

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        async with httpx.AsyncClient(timeout=300) as client:
            tasks = [infer_one(i, row, config, client, sem) for i, row in enumerate(ds)]
            from tqdm.asyncio import tqdm_asyncio

            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")

        all_metrics, examples, errors = collect_metrics(results, ds, jsonl_path)

        checkpoint_volume.commit()

        total = len(all_metrics["is_valid_json"])
        print(f"Saved {total} rows to {jsonl_path}")

        examples_table = wandb.Table(
            columns=["ground_truth", "generated"], data=examples
        )
        run.log({"examples": examples_table})

        avg_metrics = (
            {k: sum(v) / len(v) for k, v in all_metrics.items()} if total else {}
        )
        avg_metrics["total"] = total
        avg_metrics["errors"] = errors

        run.summary.update(avg_metrics)
        run.finish()

        print(f"\n{'='*60}")
        print(f"Results ({total} samples, {errors} errors):")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*60}")

        return avg_metrics

    finally:
        server.kill()
        server.wait()


@app.function(
    image=eval_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("sgai-100k")],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def run_eval(config: EvalConfig):
    return asyncio.run(run_eval_async(config))


@app.local_entrypoint()
def main(
    model_path: str = "",
    dry_run: bool = False,
    num_samples: int | None = None,
):
    if not model_path:
        raise ValueError("--model-path is required")

    config = EvalConfig(
        model_path=model_path,
        dry_run=dry_run,
        num_samples=num_samples,
    )
    print(f"Evaluating {config.model_path} on {GPU_TYPE}")
    run_eval.remote(config)
