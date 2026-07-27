import pathlib
from dataclasses import dataclass, field

import modal

MODEL_NAME = "unsloth/Qwen3-1.7B"
HF_DATASET_FINETUNING_NAME = "scrapegraphai/scrapegraph-100k-finetuning"


def build_prompt(schema: str, content: str) -> str:
    return f"""Extract data from the content according to the JSON schema.
Schema: {schema}
Content: {content}
Return ONLY valid JSON matching the schema."""


app = modal.App("scrapegraph-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "accelerate",
        "datasets",
        "hf-transfer",
        "huggingface_hub",
        "peft",
        "transformers",
        "trl",
        "unsloth[cu128-torch291] @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
        "wandb",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with train_image.imports():
    import unsloth  # noqa: F401,I001
    import datasets
    import wandb
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

model_cache_volume = modal.Volume.from_name("sg-model-cache", create_if_missing=True)
dataset_cache_volume = modal.Volume.from_name(
    "sg-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name("sg-checkpoints", create_if_missing=True)

GPU_TYPE = "A100-80GB"
TIMEOUT_HOURS = 18


@dataclass
class TrainingConfig:
    model_name: str = MODEL_NAME
    dataset_name: str = HF_DATASET_FINETUNING_NAME
    max_seq_length: int = 8192
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    batch_size: int = 8
    grad_accum: int = 2
    num_epochs: int = 2
    max_steps: int = -1
    lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    optim: str = "adamw_8bit"
    eval_split: float = 0.05
    save_steps: int = 250
    eval_steps: int = 250
    seed: int = 42
    train_on_completions: bool = True
    dry_run: bool = False
    experiment_name: str | None = None
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


def find_resume_checkpoint(output_dir: str) -> str | None:
    ckpt_dir = pathlib.Path(output_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = list(ckpt_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    print(f"Resuming from {latest_checkpoint}")
    return str(latest_checkpoint)


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("sgai-100k")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=2),
    single_use_containers=True,
)
def finetune(config: TrainingConfig):
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _format(examples):
        texts = []
        for schema, content, response in zip(
            examples["schema"], examples["content"], examples["response"], strict=True
        ):
            messages = [
                {"role": "user", "content": build_prompt(schema, content)},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
            )
            texts.append(text)
        return {"text": texts}

    cache_path = pathlib.Path("/dataset_cache") / config.dataset_name.replace("/", "--")
    if (cache_path / "train").exists():
        print("Loading cached dataset...")
        train_ds = datasets.load_from_disk(str(cache_path / "train"))
        eval_ds = datasets.load_from_disk(str(cache_path / "eval"))
    else:
        print("Downloading dataset...")
        ds = datasets.load_dataset(config.dataset_name, split="train")
        assert isinstance(ds, datasets.Dataset)
        splits = ds.train_test_split(test_size=config.eval_split, seed=config.seed)
        train_ds = splits["train"].map(
            _format, batched=True, remove_columns=ds.column_names, num_proc=4
        )
        eval_ds = splits["test"].map(
            _format, batched=True, remove_columns=ds.column_names, num_proc=4
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(cache_path / "train"))
        eval_ds.save_to_disk(str(cache_path / "eval"))
        dataset_cache_volume.commit()

    assert isinstance(train_ds, datasets.Dataset)
    assert isinstance(eval_ds, datasets.Dataset)

    max_steps = config.max_steps
    save_steps = config.save_steps
    eval_steps = config.eval_steps
    if config.dry_run:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        eval_ds = eval_ds.select(range(min(20, len(eval_ds))))
        max_steps = 20
        save_steps = 10
        eval_steps = 10

    run = wandb.init(
        entity="francesco-zuppichini",
        project="scrapegraphai-100k",
        name=config.experiment_name,
        config=config.__dict__,
        tags=["train", "qlora", "modal", "a100-80gb"],
    )
    run_identifier = run.name or run.id
    output_dir = f"/checkpoints/{run_identifier}"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # unsloth patches SFTTrainer to accept these pre-trl-0.13 arguments
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # ty: ignore[unknown-argument]
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",  # ty: ignore[unknown-argument]
        max_seq_length=config.max_seq_length,  # ty: ignore[unknown-argument]
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_epochs,
            max_steps=max_steps,
            learning_rate=config.lr,
            logging_steps=1,
            optim=config.optim,
            weight_decay=config.weight_decay,
            max_grad_norm=0.3,
            lr_scheduler_type=config.lr_scheduler,
            seed=config.seed,
            output_dir=output_dir,
            report_to="wandb",
            bf16=True,
            save_strategy="steps",
            save_steps=save_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            packing=False,
            per_device_eval_batch_size=config.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    # https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    if config.train_on_completions:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    effective_batch = config.batch_size * config.grad_accum
    print(
        f"Batch: {config.batch_size} x {config.grad_accum} = {effective_batch} effective"
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params:,}")

    resume_ckpt = find_resume_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    checkpoint_volume.commit()
    wandb.finish()
    print(f"Done. Model at {final_path}")


@app.local_entrypoint()
def main():
    config = TrainingConfig()
    effective_batch = config.batch_size * config.grad_accum
    print(
        f"Launching on {GPU_TYPE} | batch={config.batch_size}x{config.grad_accum}={effective_batch}"
    )
    finetune.remote(config)
