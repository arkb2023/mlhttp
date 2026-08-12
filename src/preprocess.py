"""Stage 1 — Preprocess CSIC 2010 HTTP request data.

Purpose
-------
Convert the raw CSIC 2010 CSV from a parameter-level representation into
a request-level representation suitable for later feature engineering.

One output row represents one logical HTTP request.

Key design decisions
--------------------
1. The raw CSV contains multiple rows per logical request.
2. The raw `index` column is not globally unique in the combined CSV.
3. The combined CSV contains two contiguous source partitions:
   - anomalous rows first
   - normal rows second
4. A globally safe request identifier is therefore:

       request_id = source_partition + "_" + index

5. Payload fragments are aggregated in original row order with "&".
6. The reconstructed payload is a modeling representation, not necessarily
   a byte-perfect reconstruction of the original HTTP request.

Output
------
Writes a request-level CSV suitable for the featurise stage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


#TASK_BASE = Path(os.environ["PROJECT_ROOT"])
TASK_BASE = Path(os.environ["PWD"]) / "../"

RAW_DEFAULT = TASK_BASE / "data" / "raw" / "output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv"
PROCESSED_DEFAULT = TASK_BASE / "data" / "processed" / "csic_2010_processed.csv"

COLUMN_RENAME_MAP = {
    "userAgent": "user_agent",
    "contentLength": "content_length",
    "contentType": "content_type",
    "cacheControl": "cache_control",
    "acceptEncoding": "accept_encoding",
    "acceptCharset": "accept_charset",
    "acceptLanguage": "accept_language",
}

REQUEST_LEVEL_COLUMNS = [
    "method",
    "url",
    "protocol",
    "user_agent",
    "pragma",
    "cache_control",
    "accept",
    "accept_encoding",
    "accept_charset",
    "accept_language",
    "host",
    "connection",
    "cookie",
    "label",
]

OPTIONAL_COLUMNS = [
    "content_length",
    "content_type",
    "payload",
]

GROUP_KEYS = ["source_partition", "index"]


# def load_raw_csv(input_path: Path) -> pd.DataFrame:
#     """Load the raw CSV."""
#     return pd.read_csv(
#         input_path,
#         dtype={"contentType": "string"},
#         low_memory=False,
#     )

def load_raw_csv(input_path: Path) -> pd.DataFrame:
    """Load the raw CSV."""
    # empty cells in the CSV are read as NaN
    # explicit NaN strings are also read as NaN
    # other strings like "NA" are not accidentally treated as missing
    return pd.read_csv(
        input_path,
        dtype={"content_type": "string"},
        low_memory=False,
        keep_default_na=False,
        na_values=[""],
    )

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename selected columns to snake_case."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def derive_source_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive source partitions from contiguous label blocks.

    Expected layout:
    - first block: anom
    - second block: norm

    This is validated from the raw file ordering.
    """
    if "label" not in df.columns:
        raise ValueError("Raw CSV must contain a 'label' column.")

    label_changed = df["label"].ne(df["label"].shift())
    transition_positions = df.index[label_changed].tolist()

    if len(transition_positions) != 2:
        raise ValueError(
            "Expected exactly two contiguous label blocks in the raw CSV, "
            f"but found {len(transition_positions)} transitions at positions "
            f"{transition_positions}."
        )

    first_row_label = df.loc[transition_positions[0], "label"]
    second_row_label = df.loc[transition_positions[1], "label"]

    if first_row_label != "anom" or second_row_label != "norm":
        raise ValueError(
            "Unexpected label block ordering. Expected first block 'anom' and "
            f"second block 'norm', got '{first_row_label}' then '{second_row_label}'."
        )

    df = df.copy()
    transition_start = transition_positions[1]
    df["source_partition"] = "anom"
    df.loc[transition_start:, "source_partition"] = "norm"
    return df


def add_request_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a globally unique request identifier."""
    df = df.copy()
    df["request_id"] = (
        df["source_partition"].astype(str) + "_" + df["index"].astype(str)
    )
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    required = set(GROUP_KEYS + REQUEST_LEVEL_COLUMNS + OPTIONAL_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def count_distinct_non_null(series: pd.Series) -> int:
    """Count distinct values ignoring nulls."""
    return series.dropna().nunique()


def validate_group_consistency(
    df: pd.DataFrame,
    group_keys: list[str],
    columns: list[str],
) -> dict[str, int]:
    """
    Validate that request-level metadata columns are consistent within each group.

    Returns:
        {column_name: number_of_groups_with_multiple_distinct_non_null_values}
    """
    inconsistencies: dict[str, int] = {}

    grouped = df.groupby(group_keys, sort=False)

    for col in columns:
        conflicting_groups = grouped[col].apply(count_distinct_non_null)
        inconsistencies[col] = int((conflicting_groups > 1).sum())

    return inconsistencies


def join_payload_fragments(values: pd.Series) -> str:
    """
    Join non-empty payload fragments in original row order using '&'.

    pandas groupby preserves row order within each group. Using sort=False on
    groupby preserves group-key appearance order as well.
    """
    fragments = values.dropna().astype(str)
    fragments = fragments[fragments.str.strip() != ""]
    if fragments.empty:
        return ""
    return "&".join(fragments.tolist())


def has_non_empty_value(values: pd.Series) -> int:
    """Return 1 if any non-null, non-empty value exists in the series, else 0."""
    non_empty = values.dropna().astype(str)
    non_empty = non_empty[non_empty.str.strip() != ""]
    return int(not non_empty.empty)


def first_non_null(values: pd.Series):
    """Return the first non-null value if present, else pd.NA."""
    non_null = values.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def aggregate_requests(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parameter-level rows into request-level rows."""
    grouped = df.groupby(GROUP_KEYS, sort=False)

    aggregated = grouped.agg(
        request_id=("request_id", "first"),
        method=("method", "first"),
        url=("url", "first"),
        protocol=("protocol", "first"),
        user_agent=("user_agent", "first"),
        pragma=("pragma", "first"),
        cache_control=("cache_control", "first"),
        accept=("accept", "first"),
        accept_encoding=("accept_encoding", "first"),
        accept_charset=("accept_charset", "first"),
        accept_language=("accept_language", "first"),
        host=("host", "first"),
        connection=("connection", "first"),
        cookie=("cookie", "first"),
        label=("label", "first"),
        content_length=("content_length", first_non_null),
        content_type=("content_type", first_non_null),
        payload=("payload", join_payload_fragments),
        has_content_length=("content_length", has_non_empty_value),
        has_content_type=("content_type", has_non_empty_value),
        has_payload=("payload", has_non_empty_value),
    ).reset_index()

    return aggregated


def fill_selected_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill selected missing values after missingness indicators are created."""
    df = df.copy()

    df["content_length"] = pd.to_numeric(df["content_length"], errors="coerce").fillna(0)
    df["content_type"] = df["content_type"].fillna("")
    df["payload"] = df["payload"].fillna("")

    return df

def validate_output(df: pd.DataFrame) -> None:
    """Validate key output invariants."""
    if df["request_id"].duplicated().any():
        dupes = int(df["request_id"].duplicated().sum())
        raise ValueError(f"Output contains duplicate request_id values: {dupes}")

    valid_labels = {"anom", "norm"}
    actual_labels = set(df["label"].dropna().unique())
    if not actual_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected labels found: {sorted(actual_labels - valid_labels)}"
        )

    for col in ["content_length", "content_type", "payload"]:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' still contains nulls after fill.")


def print_summary(
    df_raw: pd.DataFrame,
    df_agg: pd.DataFrame,
    inconsistencies: dict[str, int],
) -> None:
    """Print a compact preprocessing summary."""
    print(f"[preprocess] Raw shape: {df_raw.shape}")
    print(f"[preprocess] Aggregated shape: {df_agg.shape}")
    print(f"[preprocess] Unique request count: {df_agg['request_id'].nunique()}")

    print("[preprocess] Source partitions:")
    print(df_raw["source_partition"].value_counts())

    print("[preprocess] Label distribution:")
    print(df_agg["label"].value_counts())

    conflicting = {k: v for k, v in inconsistencies.items() if v > 0}
    if conflicting:
        print("[preprocess] WARNING: inconsistent grouped columns found:")
        for col, count in conflicting.items():
            print(f"  - {col}: {count} groups with multiple distinct non-null values")
    else:
        print("[preprocess] Group consistency check passed.")

    print("[preprocess] Null counts after fill:")
    print(df_agg[["content_length", "content_type", "payload"]].isna().sum())

    print("[preprocess] Missingness indicator sums:")
    print(
        df_agg[["has_content_length", "has_content_type", "has_payload"]].sum()
    )

    print("[preprocess] Aggregated head:")
    print(df_agg.head())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess CSIC 2010 HTTP dataset into request-level CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(RAW_DEFAULT),
        help="Path to raw CSIC CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DEFAULT),
        help="Path to write preprocessed CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] Loading raw CSV from: {input_path}")
    df_raw = load_raw_csv(input_path)

    print("[preprocess] Normalizing column names...")
    df_raw = normalize_column_names(df_raw)

    print("[preprocess] Deriving source partitions...")
    df_raw = derive_source_partition(df_raw)

    print("[preprocess] Creating request_id...")
    df_raw = add_request_id(df_raw)

    print("[preprocess] Validating required columns...")
    validate_required_columns(df_raw)

    print("[preprocess] Validating grouped column consistency...")
    inconsistencies = validate_group_consistency(
        df_raw,
        group_keys=GROUP_KEYS,
        columns=REQUEST_LEVEL_COLUMNS + ["content_length", "content_type"],
    )

    print("[preprocess] Aggregating by source-aware request key...")
    df_agg = aggregate_requests(df_raw)

    print("[preprocess] Filling selected missing values...")
    df_agg = fill_selected_missing_values(df_agg)

    print("[preprocess] Validating output invariants...")
    validate_output(df_agg)

    print_summary(df_raw, df_agg, inconsistencies)

    print(f"[preprocess] Writing preprocessed CSV to: {output_path}")
    df_agg.to_csv(output_path, index=False)

    print("[preprocess] Done.")


if __name__ == "__main__":
    main()