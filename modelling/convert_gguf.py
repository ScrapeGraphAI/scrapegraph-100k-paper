import subprocess
from pathlib import Path

import modal

MODEL_NAME = "unsloth/Qwen3-1.7B"
HF_DATASET_FINETUNING_NAME = "scrapegraphai/scrapegraph-100k-finetuning"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQ_LENGTH = 8192
CANDIDATE_POOL = 2048

LLAMA_CPP_DIR = "/app"
BIN_DIR = "/app"

app = modal.App("scrapegraph-gguf")

gguf_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:full-cuda", add_python="3.12")
    .dockerfile_commands("ENTRYPOINT []", "CMD []")
    .uv_pip_install(
        "datasets",
        "gguf",
        "hf-transfer",
        "huggingface_hub",
        "numpy",
        "peft",
        "protobuf",
        "sentencepiece",
        "torch",
        "transformers",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("modelling")
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


def convert_to_f16_gguf(model_dir: str, output_dir: str) -> str:
    out_path = f"{output_dir}/model-f16.gguf"
    if Path(out_path).exists():
        print(f"f16 GGUF already exists at {out_path}, skipping")
        return out_path

    cmd = [
        "python",
        f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
        model_dir,
        "--outfile",
        out_path,
        "--outtype",
        "f16",
    ]
    print(f"Converting to f16 GGUF: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_gb = Path(out_path).stat().st_size / 1e9
    print(f"f16 GGUF: {out_path} ({size_gb:.2f} GB)")
    return out_path


def generate_calibration_data(output_dir: str) -> str:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    cal_path = f"{output_dir}/calibration.txt"
    if Path(cal_path).exists():
        print(f"Calibration data already exists at {cal_path}, skipping")
        return cal_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset(HF_DATASET_FINETUNING_NAME, split="train")
    assert isinstance(ds, Dataset)
    ds = ds.shuffle(seed=42).select(range(CANDIDATE_POOL))

    texts = []
    for row in ds:
        messages = [
            {"role": "user", "content": build_prompt(row["schema"], row["content"])},
            {"role": "assistant", "content": row["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )
        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        if n_tokens <= MAX_SEQ_LENGTH:
            texts.append(text)
        if len(texts) >= NUM_CALIBRATION_SAMPLES:
            break

    print(
        f"Calibration samples: {len(texts)} (filtered from {CANDIDATE_POOL} candidates)"
    )

    with open(cal_path, "w") as f:
        for t in texts:
            f.write(t + "\n")

    return cal_path


def compute_imatrix(f16_path: str, cal_path: str, output_dir: str) -> str:
    imatrix_path = f"{output_dir}/imatrix.dat"
    if Path(imatrix_path).exists():
        print(f"imatrix already exists at {imatrix_path}, skipping")
        return imatrix_path

    cmd = [
        f"{BIN_DIR}/llama-imatrix",
        "-m",
        f16_path,
        "-f",
        cal_path,
        "-o",
        imatrix_path,
        "-ngl",
        "-1",
        "--chunks",
        "128",
    ]
    print(f"Computing imatrix: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"imatrix saved to {imatrix_path}")
    return imatrix_path


def quantize_gguf(
    f16_path: str, imatrix_path: str, output_dir: str, quant_type: str
) -> str:
    quant_upper = quant_type.upper()
    out_path = f"{output_dir}/model-{quant_type.lower()}.gguf"
    if Path(out_path).exists():
        print(f"{quant_type} GGUF already exists at {out_path}, skipping")
        return out_path

    cmd = [
        f"{BIN_DIR}/llama-quantize",
        "--imatrix",
        imatrix_path,
        f16_path,
        out_path,
        quant_upper,
    ]
    print(f"Quantizing {quant_upper}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_gb = Path(out_path).stat().st_size / 1e9
    print(f"{quant_upper}: {out_path} ({size_gb:.2f} GB)")
    return out_path


def upload_to_hub(output_dir: str, hub_repo: str):
    from huggingface_hub import HfApi

    api = HfApi()
    gguf_files = list(Path(output_dir).glob("*.gguf"))
    print(f"Uploading {len(gguf_files)} GGUF files to {hub_repo}")

    for f in gguf_files:
        print(f"  Uploading {f.name} ({f.stat().st_size / 1e9:.2f} GB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=hub_repo,
            repo_type="model",
        )
    print("Upload done")


@app.function(
    image=gguf_image,
    gpu=GPU_TYPE,
    cpu=8.0,
    memory=32768,
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
    quant_types: list[str],
    hub_repo: str | None,
):
    resolved = resolve_model(model_path, lora_path)
    output_dir = str(Path(resolved).parent / "gguf")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    f16_path = convert_to_f16_gguf(resolved, output_dir)
    cal_path = generate_calibration_data(output_dir)
    imatrix_path = compute_imatrix(f16_path, cal_path, output_dir)

    for qt in quant_types:
        quantize_gguf(f16_path, imatrix_path, output_dir, qt)

    if hub_repo:
        upload_to_hub(output_dir, hub_repo)

    checkpoint_volume.commit()

    print(f"\nAll done. Output dir: {output_dir}")
    for f in sorted(Path(output_dir).glob("*")):
        size = f.stat().st_size / 1e9 if f.is_file() else 0
        print(f"  {f.name}: {size:.2f} GB" if size else f"  {f.name}")


@app.local_entrypoint()
def main(
    model_path: str | None = None,
    lora_path: str | None = None,
    quant: str = "q4_k_m,q8_0",
    push_to_hub: bool = False,
    hub_repo: str = "scrapegraphai/sgai-qwen3-1.7b-gguf",
):
    quant_types = [q.strip() for q in quant.split(",")]
    repo = hub_repo if push_to_hub else None
    print(
        f"Converting to GGUF | model={model_path} lora={lora_path} quants={quant_types}"
    )
    convert.remote(model_path, lora_path, quant_types, repo)
