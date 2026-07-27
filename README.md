<p align="center">
  <img src="sgai-100k-banner.png" alt="ScrapeGraphAI Logo"/>
</p>

<h1 align="center">🕷️ ScrapeGraphAI-100k</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow" alt="HuggingFace"/></a>
  <a href="https://huggingface.co/datasets/scrapegraphai/scrapegraph-100k-finetuning"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Finetuning%20Dataset-yellow" alt="HuggingFace Finetuning"/></a>
  <a href="https://arxiv.org/abs/2602.15189"><img src="https://img.shields.io/badge/arXiv-2602.15189-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://github.com/ScrapeGraphAI/Scrapegraph-ai"><img src="https://img.shields.io/badge/GitHub-ScrapeGraphAI-blue" alt="GitHub"/></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"/>
</p>

<p align="center">
  <b>Reproducibility repository for "ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation"</b>
</p>

---

# Overview

This repository contains the code behind the [ScrapeGraphAI-100k paper](https://arxiv.org/abs/2602.15189): dataset analysis scripts, evaluation metrics, the finetuning/distillation pipeline, the PII audit, and the LLM-judge pilot.

The paper releases two datasets and a model:

- **[scrapegraphai/scrapegraphai-100k](https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k)** — 93,695 real-world schema-constrained extraction events collected via opt-in telemetry of the [ScrapeGraphAI](https://github.com/ScrapeGraphAI/Scrapegraph-ai) library (Q2–Q3 2025), derived from ~9M raw events, deduplicated and balanced for schema diversity (25,453 unique schemas, 8,155 root domains). Each row holds the prompt, Markdown-converted page content, target JSON schema, LLM response, and metadata (model, latency, five schema-complexity metrics, and a structural `response_is_valid` flag — 96.5% of responses validate, invalid ones are retained for failure-mode research).
- **[scrapegraphai/scrapegraph-100k-finetuning](https://huggingface.co/datasets/scrapegraphai/scrapegraph-100k-finetuning)** — a finetuning-ready derivative with `(schema, content, response)` triples: `train` (25,244 rows) and `test` (2,808 rows) with GPT-5-nano regenerated, schema-compliant targets (teacher pseudo-labels), plus a leak-free, human-verified `human_eval` split (100 rows) for semantic-correctness evaluation.
- **[scrapegraphai/sgai-qwen3-1.7b](https://huggingface.co/scrapegraphai/sgai-qwen3-1.7b)** — Qwen3-1.7B finetuned on the finetuning dataset.

## Quick Start

```python
from datasets import load_dataset

# Full corpus: prompts, content, schemas, responses, complexity metadata
ds = load_dataset("scrapegraphai/scrapegraphai-100k")

# Finetuning derivative: train / test / human_eval splits
ft = load_dataset("scrapegraphai/scrapegraph-100k-finetuning")
print(ft["human_eval"][0])
```

## Repository Structure

| Path | Contents |
|------|----------|
| `modelling/` | Finetuning pipeline: preprocessing, training, evaluation, metrics, LoRA merge, quantization, HF push |
| `scripts/` | Paper analysis: dataset statistics, domain diversity, frontier baselines, human benchmark, LLM-judge pilot, PII audit, figures (see [`scripts/README.md`](scripts/README.md)) |
| `figures/`, `tables/` | Generated paper assets |

## Evaluation Metrics

The `modelling/metrics.py` module provides evaluation functions for JSON extraction tasks.

### JSON Validation

Check if a JSON string is valid and complies with a schema:

```python
from modelling.metrics import json_validator

schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"]
}

result = json_validator('{"name": "John"}', schema)
# {'is_valid': True, 'is_compliant': True}
```

### Metrics

Evaluate extraction quality with `magic_metric`:

```python
from modelling.metrics import magic_metric, parse_json_remove_duplicates

pred = '{"name": "John", "age": 30, "tags": ["a", "b"]}'
true = {"name": "john", "age": 30, "tags": ["b", "a"], "city": "NYC"}

pred = parse_json_remove_duplicates(pred)
if pred is None:
    pred = {}
result = magic_metric(pred, true)
```

Returns:
| Metric | Description |
|--------|-------------|
| `key_precision` | Fraction of predicted keys that exist in ground truth |
| `key_recall` | Fraction of ground truth keys found in prediction |
| `key_f1` | Harmonic mean of precision and recall |
| `missing_keys` | Count of keys in ground truth but not in prediction |
| `extra_keys` | Count of keys in prediction but not in ground truth |
| `value_score` | Average field score (BLEU for strings, exact match for numbers/bools, set comparison for arrays) |
| `overall_bleu` | BLEU score on serialized JSON strings |

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

For local serving with vllm (installed separately due to conflicting numba/numpy pins):

```bash
uv pip install vllm
```

## Links

- 🤗 [Hugging Face Dataset](https://huggingface.co/datasets/scrapegraphai/scrapegraphai-100k)
- 🤗 [Hugging Face Dataset for Finetuning](https://huggingface.co/datasets/scrapegraphai/scrapegraph-100k-finetuning)
- 🤗 [Finetuned Model (sgai-qwen3-1.7b)](https://huggingface.co/scrapegraphai/sgai-qwen3-1.7b)
- 📄 [arXiv Paper](https://arxiv.org/abs/2602.15189)
- 🕷️ [ScrapeGraphAI Library](https://github.com/ScrapeGraphAI/Scrapegraph-ai)

## Training (Modal)

```bash
modal run modelling/train.py --dry-run
modal run --detach modelling/train.py
```

## Evaluation (Modal)

```bash
modal run modelling/evaluation.py
modal run --detach modelling/evaluation.py
modal run --detach modelling/evaluation.py --lora /checkpoints/default/final
```


## Merge LoRA

Merge the LoRA adapter into the base model for faster inference (no runtime LoRA overhead, enables CUDA graphs):

```bash
python -m modelling.merge_lora \
  --model-name Qwen/Qwen3-1.7B \
  --lora-path sg-checkpoints/efficient-frost-76/final
```

Saves to `sg-checkpoints/efficient-frost-76/merged` by default. Use `--output-path` to override.

## Quantize (Modal)

AWQ W4A16 (calibrated on the finetuning train split via llmcompressor):

```bash
modal run modelling/convert_awq.py --lora-path /checkpoints/efficient-frost-76/final
modal run modelling/convert_awq.py --model-path /checkpoints/efficient-frost-76/merged --push-to-hub
```

GGUF (llama.cpp, default quants `q4_k_m,q8_0`):

```bash
modal run modelling/convert_gguf.py --lora-path /checkpoints/efficient-frost-76/final
modal run modelling/convert_gguf.py --model-path /checkpoints/efficient-frost-76/merged --quant q4_k_m --push-to-hub
```

## Serve

Quantized model (recommended, fastest):

```bash
vllm serve sg-checkpoints/efficient-frost-76/merged-w4a16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --port 8000
```

Merged model (fp16):

```bash
vllm serve sg-checkpoints/efficient-frost-76/merged \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --port 8000
```

On a 3090 (24GB), use the quantized model with Marlin kernels:

```bash
vllm serve sg-checkpoints/efficient-frost-76/merged-w4a16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --port 8000
```

Or fp16 with `--enforce-eager` to skip CUDA graph capture:

```bash
vllm serve sg-checkpoints/efficient-frost-76/merged \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

With LoRA (without merging, slowest):

```bash
vllm serve Qwen/Qwen3-1.7B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-lora \
  --lora-modules adapter=sg-checkpoints/efficient-frost-76/final \
  --enforce-eager \
  --port 8000
```

Then:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sg-checkpoints/efficient-frost-76/merged-w4a16",
    "messages": [{"role": "user", "content": "your prompt here"}],
    "temperature": 0.0,
    "max_tokens": 4096,
    "repetition_penalty": 1.1,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Citation

```bibtex
@misc{brach2026scrapegraphai100kdatasetschemaconstrainedllm,
      title={ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation},
      author={William Brach and Francesco Zuppichini and Marco Vinciguerra and Lorenzo Padoan},
      year={2026},
      eprint={2602.15189},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2602.15189},
}
```
