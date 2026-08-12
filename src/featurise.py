"""Stage 2 — Featurise CSIC 2010 request-level data.

Purpose
-------
Transform the preprocessed CSIC 2010 request-level CSV into a model-ready
feature matrix for Objective 1.

Input
-----
A request-level CSV produced by preprocess.py, with one row per logical
HTTP request.

Output
------
Writes:
- a model-ready feature CSV with numeric columns only
- a target CSV
- a metadata CSV describing retained/dropped columns

Design principles
-----------------
1. Keep the first pass simple and interpretable.
2. Prefer explicit numeric features over opaque text vectorisation.
3. Preserve traceability columns only in metadata, not in model features.
4. Ensure no missing values remain in the output feature matrix.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

print(f"[featurise] scikit-learn version: {sklearn.__version__}")

#TASK_BASE = Path(os.environ["PROJECT_ROOT"]) / "01-http-anamoly-detection"
TASK_BASE = Path(os.environ["PWD"]) / "../"
INPUT_DEFAULT = TASK_BASE / "data" / "processed" / "csic_2010_processed.csv"
FEATURES_DEFAULT = TASK_BASE / "data" / "features" / "csic_2010_features.csv"
TARGET_DEFAULT = TASK_BASE / "data" / "features" / "csic_2010_target.csv"
METADATA_DEFAULT = TASK_BASE / "data" / "features" / "csic_2010_feature_metadata.csv"

TRACE_COLUMNS = ["source_partition", "index", "request_id", "label"]

TEXT_COLUMNS = ["url", "payload", "cookie", "host"]
CATEGORICAL_COLUMNS = ["method", "content_type"]
NUMERIC_COLUMNS = [
    "content_length",
    "has_content_length",
    "has_content_type",
    "has_payload",
]

CONSTANT_CANDIDATES = [
    "protocol",
    "user_agent",
    "pragma",
    "cache_control",
    "accept",
    "accept_encoding",
    "accept_charset",
    "accept_language",
    "connection",
]

SUSPICIOUS_PATTERNS = {
    "has_sql_select": r"select",
    "has_sql_union": r"union",
    "has_sql_drop": r"drop",
    "has_sql_insert": r"insert",
    "has_sql_delete": r"delete",
    "has_script_tag": r"<script|%3cscript",
    "has_alert": r"alert|%61%6c%65%72%74",
    "has_or_true": r"\bor\b|\|\|",
    "has_comment": r"--|/\*|\*/|%2d%2d",
    "has_path_traversal": r"\.\./|%2e%2e%2f",
}


def load_preprocessed_csv(input_path: Path) -> pd.DataFrame:
    """
    Load preprocessed request-level CSV.

    pandas may read empty CSV fields back as NaN, so explicitly restore the
    columns that are intentionally allowed to be empty strings.
    """
    df = pd.read_csv(input_path, low_memory=False)
    for col in ["content_type", "payload", "cookie", "host", "url", "method"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "source_partition",
        "index",
        "request_id",
        "method",
        "url",
        "content_length",
        "content_type",
        "payload",
        "has_content_length",
        "has_content_type",
        "has_payload",
        "label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = (df["label"] == "anom").astype(int)
    return df


def safe_len(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len()


def count_regex(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.count(pattern, flags=re.IGNORECASE)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    url = df["url"].fillna("").astype(str)
    payload = df["payload"].fillna("").astype(str)
    cookie = df["cookie"].fillna("").astype(str)
    content_type = df["content_type"].fillna("").astype(str)

    df["url_length"] = safe_len(url)
    df["payload_length"] = safe_len(payload)
    df["cookie_length"] = safe_len(cookie)
    df["content_type_length"] = safe_len(content_type)

    df["url_slash_count"] = count_regex(url, r"/")
    df["url_dot_count"] = count_regex(url, r"\.")
    df["url_question_count"] = count_regex(url, r"\?")
    df["url_ampersand_count"] = count_regex(url, r"&")
    df["url_equals_count"] = count_regex(url, r"=")
    df["url_percent_count"] = count_regex(url, r"%")
    df["url_digit_count"] = count_regex(url, r"\d")
    df["url_parameter_count"] = count_regex(url, r"[?&][^=&]+=")

    df["payload_ampersand_count"] = count_regex(payload, r"&")
    df["payload_equals_count"] = count_regex(payload, r"=")
    df["payload_percent_count"] = count_regex(payload, r"%")
    df["payload_digit_count"] = count_regex(payload, r"\d")
    df["payload_parameter_count"] = count_regex(payload, r"(^|&)[^=&]+=")
    df["payload_quote_count"] = count_regex(payload, r"'|%27|\"|%22")
    df["payload_angle_bracket_count"] = count_regex(payload, r"<|>|%3c|%3e")
    df["payload_paren_count"] = count_regex(payload, r"\(|\)|%28|%29")
    df["payload_semicolon_count"] = count_regex(payload, r";|%3b")
    df["payload_slash_count"] = count_regex(payload, r"/|%2f")
    df["payload_comment_count"] = count_regex(payload, r"--|/\*|\*/|%2d%2d")
    df["payload_percent_encoded_count"] = count_regex(payload, r"%[0-9a-fA-F]{2}")

    df["cookie_semicolon_count"] = count_regex(cookie, r";")
    df["cookie_equals_count"] = count_regex(cookie, r"=")

    df["url_has_query"] = url.str.contains(r"\?", regex=True).astype(int)
    df["payload_is_empty"] = (payload == "").astype(int)
    df["content_type_is_empty"] = (content_type == "").astype(int)

    for feature_name, pattern in SUSPICIOUS_PATTERNS.items():
        df[feature_name] = (
            payload.str.contains(pattern, case=False, regex=True)
            | url.str.contains(pattern, case=False, regex=True)
        ).astype(int)

    return df


def drop_constant_columns(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    dropped: list[str] = []

    for col in candidates:
        if col in df.columns and df[col].nunique(dropna=False) <= 1:
            df = df.drop(columns=[col])
            dropped.append(col)

    return df, dropped


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    available = [col for col in columns if col in df.columns]
    if not available:
        return df, []

    # newer versions use sparse_output=False
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=float,
        )
    except TypeError:
        # older versions accept sparse=False, 
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=float,
        )

    encoded = encoder.fit_transform(df[available])
    encoded_columns = encoder.get_feature_names_out(available)
    encoded_df = pd.DataFrame(encoded, columns=encoded_columns, index=df.index)

    df = pd.concat([df.drop(columns=available), encoded_df], axis=1)
    return df, list(encoded_columns)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(TRACE_COLUMNS + TEXT_COLUMNS + ["y"])
    return [col for col in df.columns if col not in excluded]


def finalize_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    X = X.fillna(0)

    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns remain: {non_numeric}")

    return X


def build_metadata(
    input_columns: list[str],
    output_feature_columns: list[str],
    dropped_constant_columns: list[str],
    encoded_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for col in input_columns:
        rows.append(
            {
                "column": col,
                "status": "input",
                "detail": "",
            }
        )

    for col in dropped_constant_columns:
        rows.append(
            {
                "column": col,
                "status": "dropped_constant",
                "detail": "nunique <= 1",
            }
        )

    for col in encoded_columns:
        rows.append(
            {
                "column": col,
                "status": "encoded_feature",
                "detail": "one_hot",
            }
        )

    for col in output_feature_columns:
        if col not in encoded_columns:
            rows.append(
                {
                    "column": col,
                    "status": "output_feature",
                    "detail": "",
                }
            )

    return pd.DataFrame(rows)


def print_summary(
    df_input: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    dropped_constant_columns: list[str],
) -> None:
    print(f"[featurise] Input shape: {df_input.shape}")
    print(f"[featurise] Feature matrix shape: {X.shape}")
    print(f"[featurise] Target shape: {y.shape}")
    print(f"[featurise] Positive class count (anom=1): {int(y['y'].sum())}")
    print(f"[featurise] Negative class count (norm=0): {int((1 - y['y']).sum())}")
    print(f"[featurise] Dropped constant columns: {dropped_constant_columns}")
    print("[featurise] Feature preview:")
    print(X.head())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Featurise preprocessed CSIC 2010 HTTP request data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_DEFAULT),
        help="Path to preprocessed request-level CSV.",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default=str(FEATURES_DEFAULT),
        help="Path to write model-ready feature CSV.",
    )
    parser.add_argument(
        "--target-output",
        type=str,
        default=str(TARGET_DEFAULT),
        help="Path to write target CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=str(METADATA_DEFAULT),
        help="Path to write feature metadata CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    features_output = Path(args.features_output)
    target_output = Path(args.target_output)
    metadata_output = Path(args.metadata_output)

    features_output.parent.mkdir(parents=True, exist_ok=True)
    target_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[featurise] Loading preprocessed CSV from: {input_path}")
    df = load_preprocessed_csv(input_path)

    print("[featurise] Validating input...")
    validate_input(df)

    print("[featurise] Creating binary target...")
    df = add_target(df)

    print("[featurise] Adding derived features...")
    df = add_derived_features(df)

    print("[featurise] Dropping constant columns...")
    df, dropped_constant_columns = drop_constant_columns(df, CONSTANT_CANDIDATES)

    print("[featurise] One-hot encoding categorical columns...")
    df, encoded_columns = one_hot_encode(df, CATEGORICAL_COLUMNS)

    print("[featurise] Selecting model feature columns...")
    feature_columns = select_feature_columns(df)

    print("[featurise] Finalizing feature matrix...")
    X = finalize_feature_matrix(df, feature_columns)

    y = df[["request_id", "y"]].copy()

    metadata = build_metadata(
        input_columns=list(df.columns),
        output_feature_columns=list(X.columns),
        dropped_constant_columns=dropped_constant_columns,
        encoded_columns=encoded_columns,
    )

    print_summary(df, X, y, dropped_constant_columns)

    print(f"[featurise] Writing features CSV to: {features_output}")
    X.to_csv(features_output, index=False)

    print(f"[featurise] Writing target CSV to: {target_output}")
    y.to_csv(target_output, index=False)

    print(f"[featurise] Writing metadata CSV to: {metadata_output}")
    metadata.to_csv(metadata_output, index=False)

    print("[featurise] Done.")


if __name__ == "__main__":
    main()