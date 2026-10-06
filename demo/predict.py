#!/usr/bin/env python
"""
Demo CLI: classify one HTTP request with the trained model.

Usage:
    python predict.py --url "/index.html"                    # normal
    python predict.py --url "/_vti_bin/fpcount.exe"          # IIS family hit
    python predict.py --method POST --url "/login.jsp" \
        --payload "user=admin'--" --content-length 13        # payload attack
"""

import argparse
import json
import sys
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
METRICS_PATH = REPO_ROOT / "models" / "model_metrics.json"
REGISTRY_PATH = REPO_ROOT / "data" / "url_pattern_families.json"
FEATURISED_PATH = REPO_ROOT / "data" / "csic_2010_featurised.parquet"

DEFAULT_REQUEST = {
    "method": "GET",
    "url": "/index.html",
    "host": "localhost:8080",
    "payload": "",
    "contentLength": 0,
}
def load_config():
    metrics = json.loads(METRICS_PATH.read_text())
    train_cols = metrics["train_cols"]
    assert "anomalous" not in train_cols, "target leaked into train_cols"

    registry = json.loads(REGISTRY_PATH.read_text())
    family2feat = {}
    for k in registry:
      #family2feat = family2feat | {k: f"url_has_{k}_probe"} 
      family2feat[k] = f"url_has_{k}_probe"

    #print(f"family2feat {family2feat}")
    assert set(family2feat.values()) <= set(train_cols)

    featurised = pd.read_parquet(FEATURISED_PATH)
    feature_dtypes = featurised[train_cols].dtypes.to_dict()
  
    return train_cols, registry, family2feat, feature_dtypes

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["GET", "POST", "PUT"])
    p.add_argument("--url", help="request target, e.g. /index.html")
    p.add_argument("--host", help="Host header value")
    p.add_argument("--payload", help="request body (POST/PUT)")
    p.add_argument("--content-length", type=int, dest="contentLength")
    p.add_argument("--verbose", action="store_true", help="print the feature row before prediction")

    return p.parse_args(argv)

def build_request(args) -> dict:
    """CLI overrides on top of the default normal request."""
    rec = DEFAULT_REQUEST.copy()
    for key in ("method", "url", "host", "payload", "contentLength"):
        val = getattr(args, key)
        if val is not None:
            rec[key] = val
    
    # if a full URL given, split off host and path
    if rec["url"].startswith(("http://", "https://")):
        parsed = urlparse(rec["url"])
        rec["host"] = parsed.netloc
        rec["url"] = parsed.path + ("?" + parsed.query if parsed.query else "")

    # featurise the SAME representation as the training data:
    # full absolute URL, scheme + host + path
    rec["url_full"] = f"http://{rec['host']}{rec['url']}"

    return rec

def featurise_payload(rec: dict) -> dict:
    """Payload features. Same regexes as featurise.ipynb, applied to one string."""
    payload = rec["payload"]
    content_length = rec["contentLength"]
    feats = {}

    # Count based features
    feats["payload_quote_count"] = len(re.findall(r"'|%27|\"|%22", payload, flags=re.IGNORECASE))
    feats["payload_paren_count"] = len(re.findall(r"\(|\)|%28|%29", payload, flags=re.IGNORECASE))
    feats["payload_slash_count"] = len(re.findall(r"/|%2f", payload, flags=re.IGNORECASE))
    feats["payload_parameter_count"] = len(re.findall(r"(^|&)[^=&]+=", payload, flags=re.IGNORECASE))
    feats["payload_percent_count"] = len(re.findall(r"%", payload, flags=re.IGNORECASE))
    feats["payload_percent_encoded_count"] = len(re.findall(r"%[0-9a-fA-F]{2}", payload, flags=re.IGNORECASE))

    # Binary flag based features
    feats["payload_has_alert"] = bool(re.search(r"alert|%61%6c%65%72%74", payload, flags=re.IGNORECASE))

    # Additional features
    feats["payload_length"] = len(payload)
    feats["is_empty_payload"] = int(len(payload) == 0)
    feats["is_17_to_32_bytes"] = int(17 <= content_length <= 32)

    return feats

def featurise_url(url: str, registry: dict, family2feat: dict) -> dict:
    """URL features"""
    feats = {}
    feats["url_length"] = len(url)

    for family, pattern_list in registry.items():
        #assembled_pattern = "|".join(pattern_list)
        assembled_pattern = "|".join(f"(?:{p})" for p in pattern_list)
        feats[family2feat[family]] = int(bool(re.search(assembled_pattern, url)))
    return feats

def featurise_method(method: str, train_cols: list) -> dict:
    feats = {}
    method_col = f"method_{method}"
    if method_col in train_cols:
        feats[method_col] = 1
    
    return feats   

def featurise_host(host: str, train_cols: list) -> dict:
    feats = {}    
    host_col = f"host_{host}"
    if host_col in train_cols:
        feats[host_col] = 1
    
    return feats   

def featurise_request(train_cols: list, rec: dict, registry: dict, family2feat: dict) -> dict:
    feats = {}

    feats |= featurise_payload(rec)
    #feats |= featurise_url(rec["url"], registry, family2feat)
    feats |= featurise_url(rec["url_full"], registry, family2feat)
    feats |= featurise_method(rec["method"], train_cols)
    feats |= featurise_host(rec["host"], train_cols)

    return feats

def build_row(feats: dict, train_cols: list, feature_dtypes: dict) -> pd.DataFrame:
    row = pd.DataFrame(index=[0])
    for col in train_cols:
        row[col] = pd.Series([feats.get(col, 0)], dtype=feature_dtypes[col])
    return row

def main(argv=None):
    train_cols, registry, family2feat, feature_dtypes = load_config()
    args = parse_args(argv)
    rec = build_request(args)
    print(f"request: {rec['method']} {rec['url']} (host={rec['host']})")

    feats = featurise_request(train_cols, rec, registry, family2feat)

    row = build_row(feats, train_cols, feature_dtypes)

    if args.verbose:
        print("feature row (non-zero)")
        for col, val in row.iloc[0][row.iloc[0] != 0].items():
            print(f"  {col} = {val}")

        dropped = set(feats) - set(train_cols)
        if dropped:
            print(f"WARNING: computed features not in train_cols: {sorted(dropped)}")
  
    model = joblib.load(MODEL_PATH)
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    print(f"prediction: {'anomalous' if pred == 1 else 'normal'} "
          f"(anomaly probability {proba:.4f})")

    fired = [c for c in train_cols if row.at[0, c] != 0]
    print("fired features: " + ", ".join(fired) if fired else "fired features: none")

    return pred

if __name__ == "__main__":
    main()