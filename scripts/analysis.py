import fasttext
import matplotlib.pyplot as plt
import pandas as pd
from graphs import FIGURE_FORMAT, FIGURES_DIR, TABLES_DIR, load_data
from huggingface_hub import hf_hub_download
from tqdm import tqdm

_ft_model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification",
    filename="model.bin",
)
_ft_model = fasttext.load_model(_ft_model_path)


def detect_language(text: str) -> str:
    # Use the C-level predict to avoid numpy 2.x copy=False issue
    # in the Python wrapper's predict method.
    result = _ft_model.f.predict(text.replace("\n", " "), 1, 0.0, "")
    if not result:
        return "unknown"
    _, label = result[0]
    return label.replace("__label__", "")


def language_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CONTENT LANGUAGE ANALYSIS")
    print("=" * 60)
    langs = [
        detect_language(text) if isinstance(text, str) and text.strip() else "unknown"
        for text in tqdm(df["content"], desc="Detecting languages")
    ]
    langs = pd.Series(langs)
    lang_counts = langs[langs != "unknown"].value_counts()
    total = lang_counts.sum()

    lang_stats = pd.DataFrame(
        {
            "count": lang_counts,
            "percentage": (lang_counts / total * 100).round(2),
        }
    )
    lang_stats.index.name = "language"
    lang_stats.to_csv(TABLES_DIR / "language_stats.csv")
    print(
        f"Saved language stats ({len(lang_stats)} languages) to {TABLES_DIR / 'language_stats.csv'}"
    )

    # Horizontal bar chart (top 15 + Other)
    top_n = 15
    top_langs = lang_counts.head(top_n)
    other_count = lang_counts.iloc[top_n:].sum()

    if other_count > 0:
        plot_data = pd.concat([top_langs, pd.Series({"Other": other_count})])
    else:
        plot_data = top_langs

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

    for count, bar in zip(plot_data.values, bars, strict=True):
        ax.text(
            count + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count / total * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"content_language_dist.{FIGURE_FORMAT}")
    plt.close()
    print(f"Saved content_language_dist.{FIGURE_FORMAT}")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    df = load_data()
    language_analysis(df)


if __name__ == "__main__":
    main()
