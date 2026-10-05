# HTTP Traffic Anomaly Detection
Machine learning-based anomaly detection on HTTP request traffic, built on the [CSIC 2010 dataset](http://lexr.ai/csic_dataset/output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv.zip)

## Overview

This project detects anomalous HTTP requests. The anomalies include injection
attempts, scanner probes, and malformed traffic. The method is supervised
classification.

The CSIC 2010 traffic contains 61,065 requests: 36,000 normal and 25,065
anomalous. The requests are preprocessed, featurised, and used to train a
Random Forest classifier. The selected model catches 94.5% of anomalies with
84.2% precision on a held-out test set.

The URL analysis is the distinctive part of the approach. Anomalous URLs are
mined into 15 regex pattern families. The families are verified two ways: every
anomalous URL matches some family, and no family matches any normal URL. The
verified families are frozen into a registry and become the model's URL-level
features.

## Approach

Three stages, in order:

1. **Preprocessing & EDA** — [preprocess.ipynb](./notebooks/preprocess.ipynb)
   - Input: [CSIC 2010 raw CSV](https://petescully.co.uk/research/csic-2010-http-dataset-in-csv-format-for-weka-analysis/)
   - Parses raw request lines into columns: `method`, `url`, `host`, `payload`,
     `contentLength`
   - Drops non-predictive identifiers: `cookie`, `index`
   - Output: [csic_2010_preprocessed.parquet](./data/csic_2010_preprocessed.parquet)

2. **Featurisation** — [featurise.ipynb](./notebooks/featurise.ipynb)
   - One-hot encodes `method` and `host`
   - Derives length and payload-pattern features
   - Featurisation runs in two passes, with URL analysis in between:

       **2a. URL pattern-family discovery** — [analyse_urls.ipynb](./notebooks/analyse_urls.ipynb)
       - Decomposes the URL universe: 1,615 anomaly-only URLs, 20 normal-only,
         8 seen in both classes
       - Mines the anomalous-only URLs into 15 regex pattern families
       - Verifies coverage: 1,615 of 1,615 anomalous URLs match a family, and 0
         normal URLs match.
       - Output: [url_pattern_families.json](./data/url_pattern_families.json)

       **2b. Feature building**
       - Builds 15 binary URL features from the pattern registry
       - Validates every URL feature at request level: zero hits on normal
         requests
   - Output: [csic_2010_featurised.parquet](./data/csic_2010_featurised.parquet) —
     61,065 rows × 33 columns (32 features + target `anomalous`)

3. **Training & evaluation** — [evaluate.ipynb](./notebooks/evaluate.ipynb)
   - Stratified 70:30 train/test split
   - Random Forest baseline, two trivial-classifier baselines
   - `GridSearchCV` tuning with stratified 5-fold CV
   - Final selection by held-out test F1
   - Output: [selected model](./models/model.joblib) +
     [model_metrics.json](./models/model_metrics.json)

## Repository Structure

```bash
.
├── data
│   ├── anom_get_urls_counts.csv          # Anomalous GET URLs, frequency-ranked
│   ├── anom_only_urls.parquet            # Anomaly-only URLs with label counts
│   ├── csic_2010_featurised.parquet      # Feature matrix: 61,065 × 33 (32 features + target)
│   ├── csic_2010_preprocessed.parquet    # Request-level dataset
│   ├── csic_2010_raw.csv                 # Downloaded raw dataset
│   ├── norm_only_urls.parquet            # Normal-only URLs with label counts
│   └── url_pattern_families.json         # Frozen 15-family regex registry
├── models
│   ├── model.joblib                      # Selected Random Forest model
│   └── model_metrics.json                # Model parameters, feature list, test metrics
├── notebooks
│   ├── analyse_urls.ipynb                # URL analysis
│   ├── evaluate.ipynb                    # Model training & evaluation
│   ├── featurise.ipynb                   # Feature engineering
│   └── preprocess.ipynb                  # Exploratory data analysis
└── README.md
```

## Results

Test set: `18,320` requests (`10,800` normal, `7,520` anomalous), held out from training.  

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR |  
|---|---|---|---|---|---|---|  
| RF (default) | 0.9044 | 0.9036 | 0.8588 | 0.8806 | 0.0638 | 0.1412 |  
| **RF (tuned, selected)** | **0.9046** | 0.8420 | **0.9449** | **0.8905** | 0.1234 | **0.0551** |  
| Dummy (most frequent) | 0.5895 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |  
| Constant (always anom) | 0.4105 | 0.4105 | 1.0 | 0.5820 | 1.0 | 0.0 |  

The two trivial classifiers are the operating extremes: never alert and always
alert. A useful detector must beat both F1 values (0.0 and 0.5820). Both Random
Forests do.

**Selected model:** `RandomForestClassifier(class_weight="balanced", max_depth=15, min_samples_split=10, n_estimators=100)`

- **Selection rule:** highest F1 on the held-out test set. The tuned model scores `0.8905`. The default model scores `0.8806`. The tuned model wins.
- `class_weight="balanced"` was the decisive parameter. It raises `recall` from `0.8588` to `0.9449`. Missed anomalies drop from `14.1%` to `5.5%`. The tradeoff is `precision`, which falls from `0.9036` to `0.8420`, and false alerts, which rise from `6.4%` to `12.3%`.
- For intrusion detection, a missed attack costs more than a false alert. The tuned model is the better choice here. If alert volume is the constraint, the default model is the better choice instead.
- **Cross-validation** scores the default model on `5` folds of the **training split**. The folds are the same ones the **grid search** used. The CV numbers and the held-out test numbers agree to within about `0.002` on every metric, viz. CV recall `0.8609` vs test recall `0.8588`, and CV F1 `0.8797` vs test F1 `0.8806`. This agreement means the model is not overfit, viz. it performs the same on data it trained on and on data it never saw, so the test numbers are not an artifact of one particular split.

## Limitations

- **Split granularity:** The test split is over request rows, not over URLs. Normal URLs repeat heavily in this dataset. Each normal URL appears about 1,000 times. After a row-level split, the same URL appears on both sides. The training set and the test set therefore contain near-identical rows viz. the same URL with slightly different payloads or byte counts. A model can score well on the test set partly by remembering URLs it saw during  training. The test numbers measure performance on repeated traffic. A URL-grouped split, where no URL appears on both sides, is reported in evaluate.ipynb. The two splits bracket the true performance: the row split is the optimistic end, and the grouped split is the conservative end.  Production traffic contains both old and new URLs, so the truth sits between the two numbers.
- **Lab traffic.** CSIC 2010 is synthetic traffic against one web application (an e-commerce shop). The numbers do not transfer directly to production traffic
- **Known error concentration.** Almost all residual false negatives sit on `8` endpoints whose URLs look legitimate. These `8` endpoints carry both normal and anomalous traffic. URL features cannot separate the two classes there, viz. the URL patterns of the attacks and of the normal traffic are identical, so only payload and length features can reach them. Improving `recall` further needs better payload and sequence features.


## Setup
```bash
pip install -r requirements.txt
```

Run the notebooks in order:

```bash
preprocess.ipynb      # raw CSV -> preprocessed parquet
featurise.ipynb       # pass 1: exports the URL sets
analyse_urls.ipynb    # URL sets -> pattern-family registry
featurise.ipynb       # pass 2: registry -> featurised parquet
evaluate.ipynb        # featurised parquet -> trained model + metrics
```

**References:**  
- [CSIC 2010 HTTP dataset in CSV format for Weka analysis — Peter Scully](https://petescully.co.uk/research/csic-2010-http-dataset-in-csv-format-for-weka-analysis/)

---

*This is an academic experimental project exploring ML-based HTTP traffic anomaly detection.*