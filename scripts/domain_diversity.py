import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tldextract
from huggingface_hub import hf_hub_download

TABLES_DIR = Path("tables")
TABLES_DIR.mkdir(exist_ok=True)

DATASET_REPO = "scrapegraphai/scrapegraphai-100k"
DATASET_REVISION = "v1.0"

# offline snapshot of the public suffix list bundled with tldextract
_extract = tldextract.TLDExtract(suffix_list_urls=())


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def parse_source(source: str) -> tuple[str | None, str | None]:
    """Return (hostname, root_domain) for a source URL, or (None, None) if not a
    resolvable public web URL (localhost, raw IPs, non-http schemes, etc.)."""
    if not isinstance(source, str) or not source.strip():
        return None, None
    url = source.strip()
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None, None
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or is_ip(host):
        return None, None
    ext = _extract(host)
    if not ext.suffix:  # not a valid public-suffix domain
        return None, None
    root = f"{ext.domain}.{ext.suffix}"
    return host, root


def gini(counts: np.ndarray) -> float:
    sorted_counts = np.sort(counts).astype(float)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    cumulative_share = cumulative / cumulative[-1]
    return float((n + 1 - 2 * cumulative_share.sum()) / n)


def main() -> None:
    parquet_path = hf_hub_download(
        DATASET_REPO,
        "data/train.parquet",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    df = pq.read_table(parquet_path, columns=["source", "schema_hash"]).to_pandas()
    n_total = len(df)
    print(f"Rows: {n_total:,}")

    parsed = df["source"].map(parse_source)
    df["hostname"] = parsed.str[0]
    df["root_domain"] = parsed.str[1]

    web = df.dropna(subset=["root_domain"])
    n_web = len(web)
    print(f"Rows with a resolvable public web URL: {n_web:,} ({n_web / n_total:.1%})")
    print(f"Rows excluded (localhost/IP/non-URL): {n_total - n_web:,}")

    n_urls = web["source"].nunique()
    n_hosts = web["hostname"].nunique()
    n_roots = web["root_domain"].nunique()
    n_schemas = df["schema_hash"].nunique()
    print(f"Unique URLs: {n_urls:,}")
    print(f"Unique websites (hostnames): {n_hosts:,}")
    print(f"Unique root domains (eTLD+1): {n_roots:,}")
    print(f"Unique schemas (dataset-wide): {n_schemas:,}")

    # --- Concentration among top domains ---
    domain_counts = web["root_domain"].value_counts()
    counts = domain_counts.to_numpy()
    summary_rows = [
        ("rows_total", n_total),
        ("rows_with_public_web_url", n_web),
        ("unique_urls", n_urls),
        ("unique_hostnames", n_hosts),
        ("unique_root_domains", n_roots),
        ("unique_schemas", n_schemas),
    ]
    for k in (1, 5, 10, 25, 50, 100):
        share = counts[:k].sum() / n_web
        summary_rows.append((f"top_{k}_domain_share_of_rows", round(share, 4)))
        print(f"Top {k:>3} domains cover {share:.1%} of rows")
    singletons = int((counts == 1).sum())
    domain_shares = counts / n_web
    hhi = float((domain_shares**2).sum())
    gini_domain_rows = gini(counts)
    summary_rows += [
        ("domains_with_single_row", singletons),
        ("domains_with_single_row_pct", round(singletons / n_roots, 4)),
        ("median_rows_per_domain", float(np.median(counts))),
        ("hhi_domains", round(hhi, 6)),
        ("gini_domain_rows", round(gini_domain_rows, 4)),
    ]
    print(f"Domains with exactly one row: {singletons:,} ({singletons / n_roots:.1%})")
    print(f"Gini coefficient of rows over domains: {gini_domain_rows:.3f}")

    schemas_per_domain = web.groupby("root_domain")["schema_hash"].nunique()

    top_domains = (
        domain_counts.head(50).rename_axis("root_domain").reset_index(name="rows")
    )
    top_domains["share_of_rows"] = (top_domains["rows"] / n_web).round(5)
    top_domains["unique_schemas"] = top_domains["root_domain"].map(schemas_per_domain)
    top_domains_path = TABLES_DIR / "top_domains.csv"
    top_domains.to_csv(top_domains_path, index=False)

    # --- Schema diversity vs domain diversity ---
    domains_per_schema = web.groupby("schema_hash")["root_domain"].nunique()
    n_pairs = web[["schema_hash", "root_domain"]].drop_duplicates().shape[0]
    n_web_schemas = web["schema_hash"].nunique()
    single_domain_schemas = int((domains_per_schema == 1).sum())
    single_schema_domains = int((schemas_per_domain == 1).sum())

    summary_rows += [
        ("unique_schema_domain_pairs", n_pairs),
        ("schemas_on_web_rows", int(n_web_schemas)),
        ("schemas_seen_on_one_domain", single_domain_schemas),
        (
            "schemas_seen_on_one_domain_pct",
            round(single_domain_schemas / n_web_schemas, 4),
        ),
        ("mean_domains_per_schema", round(float(domains_per_schema.mean()), 3)),
        ("median_domains_per_schema", float(domains_per_schema.median())),
        ("max_domains_per_schema", int(domains_per_schema.max())),
        ("domains_with_one_schema", single_schema_domains),
        ("domains_with_one_schema_pct", round(single_schema_domains / n_roots, 4)),
        ("mean_schemas_per_domain", round(float(schemas_per_domain.mean()), 3)),
        ("median_schemas_per_domain", float(schemas_per_domain.median())),
        ("max_schemas_per_domain", int(schemas_per_domain.max())),
    ]
    print(f"Unique (schema, domain) pairs: {n_pairs:,}")
    print(
        f"Schemas tied to a single domain: {single_domain_schemas:,} "
        f"({single_domain_schemas / n_web_schemas:.1%})"
    )
    print(
        f"Domains with a single schema: {single_schema_domains:,} "
        f"({single_schema_domains / n_roots:.1%})"
    )

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_path = TABLES_DIR / "domain_diversity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path} and {top_domains_path}")


if __name__ == "__main__":
    main()
