import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(model_name: str, lora_path: str, output_path: str | None = None):
    if output_path is None:
        output_path = str(Path(lora_path).parent / "merged")

    print(f"Loading base model: {model_name}")
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    merge(args.model_name, args.lora_path, args.output_path)
