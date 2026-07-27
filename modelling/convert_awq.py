from pathlib import Path

import modal

MODEL_NAME = "unsloth/Qwen3-1.7B"
HF_DATASET_FINETUNING_NAME = "scrapegraphai/scrapegraph-100k-finetuning"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQ_LENGTH = 8192
CANDIDATE_POOL = 2048

app = modal.App("scrapegraph-awq")

awq_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "datasets",
        "hf-transfer",
        "huggingface_hub",
        "llmcompressor",
        "peft",
        "torch",
        "transformers",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

model_cache_volume = modal.Volume.from_name("sg-model-cache", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("sg-checkpoints", create_if_missing=True)

GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 2


def build_prompt(schema: str, content: str) -> str:
    return f"""Extract data from the content according to the JSON schema.
Schema: {schema}
Content: {content}
Return ONLY valid JSON matching the schema."""


def resolve_model(model_path: str | None, lora_path: str | None) -> str:
    if model_path:
        return model_path

    if not lora_path:
        raise ValueError("Either --model-path or --lora-path required")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = str(Path(lora_path).parent / "merged")
    if Path(output).exists():
        print(f"Merged model already exists at {output}, skipping merge")
        return output

    print(f"Merging LoRA from {lora_path}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, lora_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Merged to {output}")
    return output


def build_calibration_dataset(tokenizer):
    from datasets import Dataset, load_dataset

    ds = load_dataset(HF_DATASET_FINETUNING_NAME, split="train")
    assert isinstance(ds, Dataset)
    ds = ds.shuffle(seed=42).select(range(CANDIDATE_POOL))

    def preprocess(example):
        messages = [
            {
                "role": "user",
                "content": build_prompt(example["schema"], example["content"]),
            },
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        return {"text": text, "n_tokens": n_tokens}

    ds = ds.map(preprocess)
    ds = ds.filter(lambda x: x["n_tokens"] <= MAX_SEQ_LENGTH)
    ds = ds.select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
    print(f"Calibration samples: {len(ds)} (filtered from {CANDIDATE_POOL} candidates)")
    return ds


def upload_to_hub(output_dir: str, hub_repo: str):
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id=hub_repo,
        repo_type="model",
    )
    print(f"Uploaded to {hub_repo}")


@app.function(
    image=awq_image,
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=65536,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("sgai-100k")],
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def convert(
    model_path: str | None,
    lora_path: str | None,
    hub_repo: str | None,
):
    import torch
    from llmcompressor import oneshot  # ty: ignore[unresolved-import]
    from llmcompressor.modifiers.awq import (  # ty: ignore[unresolved-import]
        AWQMapping,
        AWQModifier,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_model(model_path, lora_path)
    output_dir = str(Path(resolved).parent / "awq-w4a16")

    if Path(output_dir).exists() and list(Path(output_dir).glob("*.safetensors")):
        print(f"AWQ model already exists at {output_dir}, skipping quantization")
    else:
        print(f"Loading model from {resolved}")
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved)

        ds = build_calibration_dataset(tokenizer)

        recipe = AWQModifier(
            targets="Linear",
            scheme="W4A16_ASYM",
            ignore=["lm_head"],
            offload_device=torch.device("cpu"),
            mappings=[
                AWQMapping(
                    "re:.*input_layernorm",
                    ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                ),
                AWQMapping(
                    "re:.*post_attention_layernorm", ["re:.*gate_proj", "re:.*up_proj"]
                ),
                AWQMapping("re:.*up_proj", ["re:.*down_proj"]),
            ],
        )

        print("Quantizing W4A16_ASYM AWQ...")
        oneshot(
            model=model,
            recipe=recipe,
            tokenizer=tokenizer,
            dataset=ds,
            text_column="text",
            max_seq_length=MAX_SEQ_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=output_dir,
            pipeline="sequential",
        )
        print(f"Saved to {output_dir}")

    if hub_repo:
        upload_to_hub(output_dir, hub_repo)

    checkpoint_volume.commit()

    total_size = sum(
        f.stat().st_size for f in Path(output_dir).iterdir() if f.is_file()
    )
    print(f"\nDone. Output: {output_dir} ({total_size / 1e9:.2f} GB)")


@app.local_entrypoint()
def main(
    model_path: str | None = None,
    lora_path: str | None = None,
    push_to_hub: bool = False,
    hub_repo: str = "scrapegraphai/sg-qwen3-1.7b-awq",
):
    repo = hub_repo if push_to_hub else None
    print(f"AWQ W4A16 | model={model_path} lora={lora_path}")
    convert.remote(model_path, lora_path, repo)
