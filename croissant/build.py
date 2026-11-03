# /// script
# requires-python = ">=3.11"
# dependencies = ["mlcroissant>=1.0.17", "requests", "GitPython", "pandas", "pyarrow"]
# ///
"""Build NeurIPS 2026 E&D Croissant files: HF's auto-generated core metadata + our RAI fields.

    uv run croissant/build.py              # build + validate metadata
    uv run croissant/build.py --records    # also load one record per record set (downloads parquet)
"""

import argparse
import json
from pathlib import Path

import requests

HERE = Path(__file__).parent
ORG = "scrapegraphai"

DATASETS = {
    "scrapegraphai-100k": {
        "version": "1.0.0",
        "datePublished": "2025-12-12",
        "description": (
            "93,695 real-world schema-constrained extraction events from opt-in telemetry of the "
            "ScrapeGraphAI open-source library (Q2-Q3 2025), deduplicated and balanced from ~9M raw "
            "events. Each row pairs a user prompt, a user-defined JSON schema, Markdown-converted page "
            "content and the LLM response, with model, latency, five schema-complexity metrics and a "
            "structural validity flag (jsonschema-rs). 25,453 unique schemas, 8,155 root domains."
        ),
    },
    "scrapegraph-100k-finetuning": {
        "version": "1.1.0",
        "datePublished": "2026-01-20",
        "description": (
            "Fine-tuning derivative of ScrapeGraphAI-100k v1.0: filtered, chunked (~4,096 tokens) "
            "schema-constrained extraction examples with GPT-5-nano regenerated targets "
            "(train 25,244, test 2,808) plus a 100-example human-verified human_eval split."
        ),
    },
}

SGAI = {"@type": "Organization", "name": "ScrapeGraphAI", "url": "https://scrapegraphai.com"}
STU = {"@type": "Organization", "name": "Slovak University of Technology in Bratislava", "url": "https://www.stuba.sk"}
AUTHORS = [("William Brach", STU), ("Francesco Zuppichini", SGAI), ("Marco Vinciguerra", SGAI), ("Lorenzo Padoan", SGAI)]

CITATION = """@misc{brach2026scrapegraphai100kdatasetschemaconstrainedllm,
  title={ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation},
  author={William Brach and Francesco Zuppichini and Marco Vinciguerra and Lorenzo Padoan},
  year={2026},
  eprint={2602.15189},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2602.15189}
}"""

# [Agent] The NeurIPS checker (huggingface.co/spaces/JoaquinVanschoren/croissant-rai-checker) looks for these
# exact top-level keys, so we fail the build here instead of finding out on OpenReview.
REQUIRED_RAI = [
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
    "prov:wasDerivedFrom",
    "prov:wasGeneratedBy",
]


def build(name: str, meta: dict) -> dict:
    url = f"https://huggingface.co/api/datasets/{ORG}/{name}/croissant"
    croissant = requests.get(url, timeout=60).json()
    rai = json.loads((HERE / "rai" / f"{name}.json").read_text())

    croissant["@context"] |= {"rai": "http://mlcommons.org/croissant/RAI/", "prov": "http://www.w3.org/ns/prov#"}
    croissant |= {
        "description": meta["description"],
        "version": meta["version"],
        "datePublished": meta["datePublished"],
        # [Agent] HF emits a choosealicense.com URL; the NeurIPS guidelines ask for an SPDX one.
        "license": "https://spdx.org/licenses/Apache-2.0.html",
        "creator": [{"@type": "Person", "name": name, "affiliation": org} for name, org in AUTHORS] + [SGAI],
        "publisher": {"@type": "Organization", "name": "ScrapeGraphAI", "url": "https://huggingface.co/scrapegraphai"},
        "citeAs": CITATION,
        "sameAs": ["https://arxiv.org/abs/2602.15189", "https://github.com/ScrapeGraphAI/scrapegraph-100k-paper"],
    }
    croissant |= rai

    missing = [k for k in REQUIRED_RAI if k not in croissant]
    assert not missing, f"{name}: missing RAI fields {missing}"
    return croissant


def validate(path: Path, records: bool) -> None:
    import mlcroissant as mlc

    dataset = mlc.Dataset(jsonld=str(path))
    if records:
        for record_set in dataset.metadata.record_sets:
            next(iter(dataset.records(record_set=record_set.uuid)))
            print(f"  record set {record_set.uuid}: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="store_true", help="load one record per record set")
    args = parser.parse_args()

    for name, meta in DATASETS.items():
        path = HERE / f"{name}.croissant.json"
        path.write_text(json.dumps(build(name, meta), indent=2, ensure_ascii=False) + "\n")
        validate(path, args.records)
        print(f"{path.relative_to(HERE.parent)}: valid")


if __name__ == "__main__":
    main()
