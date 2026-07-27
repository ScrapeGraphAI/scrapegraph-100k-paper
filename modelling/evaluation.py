import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import modal


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-4B"
    # model_name: str = "Qwen/Qwen4-1.7B"
    lora_path: str | None = None
    dataset_name: str = "scrapegraphai/scrapegraph-100k-finetuning"
    max_seq_length: int = 8192
    max_new_tokens: int = 2048 * 2
    temperature: float = 0.0
    dry_run: bool = False
    wandb_entity: str = "francesco-zuppichini"
    wandb_project: str = "scrapegraphai-100k"


CONFIG = EvalConfig()

app = modal.App("scrapegraph-eval")

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "datasets",
        "hf-transfer",
        "huggingface_hub",
        "jsonschema_rs",
        "orjson",
        "sacrebleu",
        "vllm",
        "wandb",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("modelling")
)

with eval_image.imports():
    import os

    import wandb
    from datasets import Dataset, load_dataset
    from vllm import LLM, SamplingParams  # ty: ignore[unresolved-import]
    from vllm.lora.request import LoRARequest  # ty: ignore[unresolved-import]

    from modelling import metrics as M
    from modelling import prompts
    from modelling.utils import extract_json

model_cache_volume = modal.Volume.from_name("sg-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("sg-checkpoints", create_if_missing=True)

TIMEOUT_HOURS = 8

GPU_TYPE = "A100-80GB"


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
    llm = LLM(
        model=config.model_name,
        max_model_len=config.max_seq_length + config.max_new_tokens,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        enable_lora=config.lora_path is not None,
    )

    lora_request = (
        LoRARequest("adapter", 1, config.lora_path) if config.lora_path else None
    )

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        job_type="eval",
        tags=["eval"],
        config=asdict(config),
    )

    ds = load_dataset(config.dataset_name, split="test")
    assert isinstance(ds, Dataset)

    if config.dry_run:
        ds = ds.select(range(min(100, len(ds))))

    all_messages = []
    valid_indices = []
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_new_tokens,
        stop=[tokenizer.eos_token],
        repetition_penalty=1.1,
    )
    max_prompt_len = config.max_seq_length

    for i, row in enumerate(ds):
        msgs = [
            {"role": "user", "content": prompts.build(row["schema"], row["content"])}
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        token_len = len(prompt_tokens)
        if token_len <= max_prompt_len:
            all_messages.append(msgs)
            valid_indices.append(i)

    print(
        f"Kept {len(valid_indices)}/{len(ds)} samples (skipped {len(ds) - len(valid_indices)} too long)"
    )

    outputs = llm.chat(
        all_messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        lora_request=lora_request,
    )

    all_metrics = defaultdict(list)
    examples = []

    run_identifier = run.name or run.id
    results_dir = "/checkpoints/results"
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = f"{results_dir}/{run_identifier}.jsonl"
    open(jsonl_path, "w").close()

    for i, (idx, output) in enumerate(zip(valid_indices, outputs, strict=True)):
        row = ds[idx]
        response = output.outputs[0].text
        schema_obj = json.loads(row["schema"])
        ground_truth = json.loads(row["response"])
        clean_response = extract_json(response)
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

        if i < 32:
            examples.append([json.dumps(ground_truth, indent=2), clean_response])

        run.log(sample_metrics, step=i)

    checkpoint_volume.commit()
    print(f"Saved {len(valid_indices)} rows to {jsonl_path}")

    examples_table = wandb.Table(columns=["ground_truth", "generated"], data=examples)
    run.log({"examples": examples_table})

    skipped = len(ds) - len(valid_indices)
    total = len(all_metrics["is_valid_json"])
    avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    avg_metrics["total"] = total
    avg_metrics["skipped"] = skipped

    run.summary.update(avg_metrics)

    run.finish()

    print(f"\n{'='*60}")
    print(f"Results ({total} samples):")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'='*60}")

    return avg_metrics


@app.local_entrypoint()
def main(
    model: str = CONFIG.model_name,
    lora: str | None = CONFIG.lora_path,
    dry_run: bool = False,
):
    config = EvalConfig(
        model_name=model,
        lora_path=lora if lora else None,
        dry_run=dry_run,
    )
    print(f"Evaluating {config.model_name} (lora={config.lora_path}) on {GPU_TYPE}")
    run_eval.remote(config)
