# MLHTTP — HTTP Traffic Anomaly Detection

Machine learning-based anomaly detection on HTTP request traffic, built on the [CSIC 2010** dataset](https://huggingface.co/datasets/arbk2023/csic_2010/resolve/main/output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv) (223,585 raw rows => 61,065 HTTP requests).

## Overview

This project detects anomalous HTTP requests (e.g., injection attacks, malformed traffic) using a supervised classification approach. Raw CSIC 2010 traffic was preprocessed, featurised, and used to train a Random Forest classifier.

**Key results (Random Forest, stratified 70/30 split):**

| Metric | Score |
|---|---|
| Accuracy | 0.8834 |
| Precision | 0.8981 |
| Recall | 0.8074 |
| F1-Score | 0.8504 |
| ROC-AUC | 0.9517 |

## Repository Structure

```
.
├── data/            # Raw, preprocessed, and featurised datasets
├── models/          # Trained models
├── notebooks/
│   ├── eda_and_analysis.ipynb  # Exploratory data analysis
│   ├── featurize.ipynb         # Feature engineering
│   └── evaluate.ipynb          # Model training & evaluation
├── env.local              # Local environment configuration
└── README.md
```

## Pipeline

1. **Raw data** → `csic_2010_raw.csv`
2. **Preprocessing** → `csic_2010_preprocessed.csv/.parquet` (cleaned, normalized fields)
3. **Featurisation** → `csic_2010_featurised.parquet` (19 numeric features)
4. **Training & evaluation** → `model.joblib` + metrics report

## Pipeline

### 1. **Raw dataset Analysis:**    


### 2. **Feature Engineering:**  
Analyse each column to derive numerical and categorical features used in model training. The features that distinguish anomalous requests from normal requests. Analyse the values per column to check for variability and characterestics that could distinguish between normal (norm) and anomalous (anom) requests.

Key Findings:  
- The `index` column is non-unique and reused across requests so it is not useful in modeling. So we drop the `index` column from the Feature set
- The `cookie` column has all cookies with length of `43` characters and a consistent format. This implies that the cookie column does not provide any variability that could distinguish between normal (norm) and anomalous (anom) requests. So we drop the `cookie` Column from the Feature set
- The `method` column provides insight that majority are `GET` `43,088` followed by `POST` `17,580` and `PUT` `397` which can influence anomaly detection. So the method column is transformed into three binary features - `method_GET`, `method_POST`, `method_PUT`. This allows the model to interpret the request type as numerical input while preserving the categorical relationships.
- The `host` column identifies the `server:port` for the request. The dataset contains only two host values `localhost:8080` and `localhost:9090`. Majority are `localhost:8080` `60,668` followed by `localhost:9090` `397`. This is useful for filtering or grouping requests by host. The `host` column is highly imbalanced, which may limit its predictive power. The host column is transformed into two binary features - `host_localhost:8080` and `host_localhost:9090`  
- The `url` column is analysed for length, count of special chars, sql patterns, suspicuous patterns and its corresponding classification (i.e anomaly or normal) as indicated in `label` column. It was found that,
    - Special Characters `/` and `.` are too common and do not help distinguish anomalies whereas `?`, `&`, `=`, `%` do not appear in the dataset, so they provide no signal. So we exclude all special character counts columns from the model
    - SQL Patterns `select` and `drop` are present and classified as anomalous. Whereas `union`, `insert`, `delete` do not appear in the dataset. So we include `select` and `drop` as binary features to for detecting SQL injection attempts.
    - Other Suspicious Patterns such as `or`, `script`, `alert`, `comment`, `path traversal`, `parameter` have zero occurrences in the dataset. `Trailing digits` appears in both anom and norm URLs and is not discriminative. `Numeric ID` is a strong indicator of anomalies. All of its occurrences are in anom URLs. So we include `numeric_id` as a binary feature and exclude all other patterns from the model.

- The `payload` column shows patterns of form submissions, login attempts, data entries etc. Typically formatted as `key-value` pairs. We analysed the values for common structure, encoded chars, sql keywords, long numeric string, suspicious patterns etc and its corresponding classification (i.e anomaly or normal) as indicated in `label` column. it was found that,
    - Special Characters: `&` no occurances and `=` is too common. Both these are not useful. `%` is also common but may be useful when combined with `percent_encoded` pattern. So we exclude `&` and `=`. and use `%` as a feature
    - SQL Patterns: No occurances of `select` `union`, `delete`. Where as `insert` appears in both `anom` and `norm` but is more common in `anom` (56% of occurrences are anomalous). `drop` appears only in `norm`, so it is not useful for anomaly detection. So we include `insert` as binary feature and exclude all other SQL patterns.  
    - In other Suspicious Patterns, `>90%` of `quote` `paranthesis` `slash` `alert` are classified as anomalous. `>50%` of `Parameter` `percent_encoded` and `insert` are anomalous. Rest all give almost no signals for anomalous classification. So we Include `quote` `paranthesis` `slash` `alert` `Parameter` `percent_encoded` `insert` and drop the rest

- The `contentLength` column analysis show that 
    - There is coorelation between increase or decrease in contentLength to anomaly or normal. Small payloads (0-10 bytes) and very large payloads (>200 bytes) have lower anomalies percentages (~40%). Medium payloads (50-200 bytes) have relatively higher anomaly percentage (~66%).
    - 100% of payloads in this range of 17-32 bytes are anomalous. However the absolute number of requests `638` are very less compared to the total. `64.19%` of payloads that are >108 bytes are anomalous. Other ranges are lesser anomalous percentages. Based on the analysis of `contentLength`, it was concluded that no distinct or consistent pattern emerged that could reliably distinguish between anomalous and normal payloads. For the current iteration, we proceed with the extracted features `is_17_to_32_bytes` and `is_empty_payload` in addition to the original `contentLength` column which already present in the dataset.

- The `label` column which is the target column contains `norm`: 36000 and `anom`: 25065 values. We created a binary target column `anomalous` with `1` for `anom` and `0` for `norm`

Detailed analyis can be found here: [featurize.ipynb](./notebooks/featurize.ipynb)

### 3. **Training & evaluation**  
The training and target feature columns are identified  
The dataset is split into Training and Test data subsets in ratio `70:30`  
The baseline model is trained using `RandomForestClassifier`
The Feature Importances are found out from the model
The model is used to make predictions on the test data subset `30%`  
The model performance evaluated by comparing model predictions against the actual targets and deriving key metrics    
The model performance is compared against DummyClassifier and ConstantClassifier model predictions metrics. This guage the effectiveness of the baseline trained model
Cross Validation is used to further evaluate the model performance.

## Features

Engineered features include:
- **Length-based**: `url_length`, `payload_length`, `contentLength`
- **URL patterns**: `url_has_numeric_id`, `url_has_select`, `url_has_drop`
- **Payload structure**: quote/paren/slash/parameter/percent-encoding counts
- **HTTP metadata**: one-hot encoded method (`GET`/`POST`/`PUT`) and host
- **Derived flags**: `is_17_to_32_bytes`, `is_empty_payload`

**Feature importance analysis** showed `payload_length` (0.286) and `url_length` (0.275) dominate predictions, followed by percent-encoding counts.

## Getting Started

```bash
# Clone and set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull versioned data
dvc pull

# Explore the notebooks
jupyter notebook notebooks/
```

## Key Findings

- The Random Forest model **reliably detects ~81% of anomalies** with ~90% precision.
- **False negatives (~19%)** are the primary limitation — flagged for future work.
- `contentLength` alone showed **no solid anomaly pattern**; length-based signals come primarily from URL and payload lengths.

## Future Work

- Advanced feature engineering
- Class rebalancing (`class_weight="balanced"`, SMOTE) to improve recall
- Hyperparameter tuning (`GridSearchCV`)
- Model comparison (XGBoost, gradient boosting, SVM, Logistic Regression)

## License

For academic and experimental use.

---

*This is an academic experimental project exploring ML-based HTTP traffic anomaly detection.*