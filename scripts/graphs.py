import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DatasetDict, load_dataset

DATASET_REPO = "scrapegraphai/scrapegraphai-100k"
DATASET_REVISION = "v1.0"
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

NUMERIC_COLS = [
    "execution_time",
    "response_size",
    "schema_size",
    "schema_depth",
    "schema_keys",
    "schema_elements",
    "schema_cyclomatic_complexity",
    "schema_complexity_score",
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.{FIGURE_FORMAT}")
    plt.close()
    print(f"Saved figures/{name}.{FIGURE_FORMAT}")


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLES_DIR / f"{name}.csv")
    print(f"Saved tables/{name}.csv")


def load_data() -> pd.DataFrame:
    ds = load_dataset(DATASET_REPO, revision=DATASET_REVISION)
    assert isinstance(ds, DatasetDict)
    df = ds["train"].to_pandas()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Dataset is empty or failed to load.")
    print(f"Loaded {df.shape[0]:,} rows with {len(df.columns)} columns")
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    stats = df[NUMERIC_COLS].describe().T
    stats["median"] = df[NUMERIC_COLS].median()
    stats = stats[["count", "mean", "std", "min", "25%", "median", "75%", "max"]]
    save_table(stats, "descriptive_stats")


def schema_complexity_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    panels = [
        (df["schema_keys"], "Number of Schema Keys"),
        (df["schema_size"] / 1024, "Schema Size (KB)"),
        (df["schema_complexity_score"], "Schema Complexity Score"),
    ]
    # Depth is small-integer valued, so a bar per value instead of a histogram
    depth_counts = df["schema_depth"].value_counts().sort_index()
    axes[0].bar(
        depth_counts.index,
        depth_counts.values,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xlabel("Schema Depth")
    axes[0].set_ylabel("Count (log scale)")
    for ax, (values, xlabel) in zip(axes[1:], panels, strict=True):
        ax.hist(values, bins=50, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlabel(xlabel)
    for ax in axes:
        ax.set_yscale("log")
    save_figure("schema_complexity_combined")

    percentiles = [25, 50, 75, 90, 95]
    metrics = [
        "schema_depth",
        "schema_keys",
        "schema_elements",
        "schema_cyclomatic_complexity",
        "schema_complexity_score",
    ]
    percentile_df = pd.DataFrame(
        {
            metric: {
                "mean": df[metric].mean(),
                "std": df[metric].std(),
                **{f"P{p}": df[metric].quantile(p / 100) for p in percentiles},
            }
            for metric in metrics
        }
    ).T
    save_table(percentile_df, "schema_percentiles")


def response_analysis(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        df["response_size"] / 1024,
        bins=50,
        color="coral",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Response Size (KB)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    save_figure("response_size_dist")


def model_performance_analysis(df: pd.DataFrame) -> None:
    model_counts = df["llm_model"].value_counts()
    top_n = 10
    plot_data = model_counts.head(top_n)
    other_count = model_counts.iloc[top_n:].sum()
    if other_count > 0:
        plot_data = pd.concat([plot_data, pd.Series({"Other": other_count})])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        range(len(plot_data)),
        plot_data.values,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data.index)
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    total = len(df)
    for count, bar in zip(plot_data.values, bars, strict=True):
        ax.text(
            count + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count / total * 100:.1f}%",
            va="center",
            fontsize=9,
        )
    save_figure("llm_model_dist")

    model_stats = (
        df.groupby("llm_model")
        .agg(
            {
                "response_is_valid": ["sum", "count", "mean"],
                "execution_time": ["mean", "median", "std"],
            }
        )
        .round(3)
    )
    model_stats.columns = [
        "valid_count",
        "total_count",
        "success_rate",
        "exec_time_mean",
        "exec_time_median",
        "exec_time_std",
    ]
    model_stats = model_stats[model_stats["total_count"] >= 100].sort_values(
        "total_count", ascending=False
    )
    save_table(model_stats, "model_stats")


def validation_vs_complexity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    overall_mean = df["response_is_valid"].mean()

    depth_validation = df.groupby("schema_depth")["response_is_valid"].mean()
    axes[0].plot(
        depth_validation.index,
        depth_validation.values,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
    )
    axes[0].set_xlabel("Depth", labelpad=15)
    axes[0].set_ylabel("Validation Rate")
    axes[0].axhline(
        y=overall_mean, color="red", linestyle="--", alpha=0.5, label="Overall Mean"
    )
    axes[0].legend(fontsize=12)

    # Bins reused by frontier_baselines.py — keep in sync
    binned_panels = [
        (
            "Keys",
            pd.cut(
                df["schema_keys"],
                bins=[0, 10, 20, 30, 50, 100, 200, float("inf")],
                labels=["1-10", "11-20", "21-30", "31-50", "51-100", "101-200", "200+"],
            ),
        ),
        (
            "Size",
            pd.cut(
                df["schema_size"],
                bins=[0, 500, 1000, 2000, 5000, 10000, float("inf")],
                labels=["<0.5KB", "0.5-1KB", "1-2KB", "2-5KB", "5-10KB", "10KB+"],
            ),
        ),
        (
            "Complexity Score",
            pd.cut(
                df["schema_complexity_score"],
                bins=[0, 50, 100, 200, 500, 1000, float("inf")],
                labels=["<50", "50-100", "100-200", "200-500", "500-1k", "1k+"],
            ),
        ),
    ]
    for ax, (xlabel, bins) in zip(axes[1:], binned_panels, strict=True):
        rate = df.groupby(bins, observed=True)["response_is_valid"].mean()
        ax.bar(range(len(rate)), rate.values, color="steelblue", edgecolor="black")
        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels(rate.index, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.axhline(y=overall_mean, color="red", linestyle="--", alpha=0.5)
    for ax in axes:
        ax.set_ylim(0, 1.05)
    save_figure("validation_vs_complexity")


def correlation_analysis(df: pd.DataFrame) -> None:
    cols = NUMERIC_COLS + ["response_is_valid"]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_figure("correlation_matrix")
    save_table(corr_matrix, "correlation_matrix")


def length_percentiles(df: pd.DataFrame) -> None:
    rows = []
    for field in ["content", "schema", "response"]:
        lengths = df[field].str.len()
        rows.append(
            {
                "field": field,
                "min": int(lengths.min()),
                **{
                    f"P{p}": int(lengths.quantile(p / 100))
                    for p in [25, 50, 75, 90, 95]
                },
                "max": int(lengths.max()),
                "mean": round(lengths.mean(), 1),
            }
        )
    result = pd.DataFrame(rows).set_index("field")
    print(result.to_string())
    save_table(result, "length_percentiles")


def find_appendix_example(df: pd.DataFrame) -> None:
    valid = df[df["response_is_valid"]].copy()
    median_complexity = valid["schema_complexity_score"].median()
    valid["complexity_dist"] = (
        valid["schema_complexity_score"] - median_complexity
    ).abs()

    content_len = valid["content"].str.len()
    p25, p75 = content_len.quantile(0.25), content_len.quantile(0.75)
    valid = valid[(content_len >= p25) & (content_len <= p75)]
    example = valid.sort_values("complexity_dist").iloc[0]

    content = example["content"]
    if len(content) > 5000:
        content = content[:5000] + "\n... [truncated]"
    output = {
        "schema": example["schema"],
        "content": content,
        "response": example["response"],
    }
    out_path = TABLES_DIR / "appendix_example.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(
        f"Saved {out_path} (complexity_score={example['schema_complexity_score']:.1f})"
    )


def model_scaling_plot() -> None:
    # Hardcoded results from the evaluation table
    sizes = [1.7, 4, 30]
    bleu_scores = [0.4581, 0.5314, 0.5935]
    ours_bleu = 0.5759

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(
        sizes,
        bleu_scores,
        s=100,
        color="coral",
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        [1.7],
        [ours_bleu],
        s=300,
        color="coral",
        marker="*",
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )

    model_names = ["Qwen3\n1.7B", "Qwen3\n4B", "Qwen3\n30B"]
    for size, score, name in zip(sizes, bleu_scores, model_names, strict=True):
        ax.text(
            size,
            score - 0.015,
            name,
            fontsize=9,
            ha="center",
            va="top",
            color="dimgray",
        )

    # Gain arrow: Qwen3-1.7B up to Ours
    ax.annotate(
        "",
        xy=(1.7, ours_bleu - 0.008),
        xytext=(1.7, bleu_scores[0] + 0.008),
        arrowprops={"arrowstyle": "->, head_width=0.3", "color": "black", "lw": 1.4},
        zorder=5,
    )
    pct_gain = (ours_bleu - bleu_scores[0]) / bleu_scores[0] * 100
    ax.text(
        1.85,
        (ours_bleu + bleu_scores[0]) / 2,
        f"+{pct_gain:.1f}%",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Gap arrow: Ours across to 30B
    ax.annotate(
        "",
        xy=(30, bleu_scores[2] - 0.005),
        xytext=(1.7, ours_bleu + 0.005),
        arrowprops={
            "arrowstyle": "->, head_width=0.3",
            "color": "gray",
            "lw": 1.0,
            "ls": "--",
        },
        zorder=2,
    )
    pct_gap = (bleu_scores[2] - ours_bleu) / bleu_scores[2] * 100
    ax.text(
        np.sqrt(1.7 * 30),  # geometric midpoint on log scale
        (ours_bleu + bleu_scores[2]) / 2 + 0.008,
        f"–{pct_gap:.1f}%",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="dimgray",
    )
    ax.text(
        1.85,
        ours_bleu + 0.01,
        "Ours (1.7B)",
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["1.7B", "4B", "30B"])
    ax.minorticks_off()
    ax.set_xlabel("Model Size (parameters)")
    ax.set_ylabel("Overall BLEU")
    ax.set_ylim(0.38, 0.65)
    save_figure("model_scaling")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    df = load_data()
    descriptive_stats(df)
    schema_complexity_analysis(df)
    response_analysis(df)
    model_performance_analysis(df)
    validation_vs_complexity(df)
    correlation_analysis(df)
    length_percentiles(df)
    find_appendix_example(df)
    model_scaling_plot()


if __name__ == "__main__":
    main()
