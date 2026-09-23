# HTTP Traffic Anomaly Detection

Machine learning-based anomaly detection on HTTP request traffic, built on the [CSIC 2010 dataset](http://lexr.ai/csic_dataset/output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv.zip) 

## Overview

This project detects anomalous HTTP requests (e.g., injection attacks, malformed traffic) using a supervised classification approach. Raw CSIC 2010 traffic was preprocessed, featurised, and used to train a Random Forest classifier.


## Approach  
1. **"Preprocessing & EDA**  
    - **Input:** [CSIC 2010 dataset](https://huggingface.co/datasets/arbk2023/csic_2010/resolve/main/output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv)  
    - **Processing notebook:** [preprocess.ipynb](./notebooks/preprocess.ipynb)
    - **Output:** Processed dataset [csic_2010_preprocessed.parquet](./data/csic_2010_preprocessed.parquet)  
2. **Featurisation**
    - **Input:** Preprocessed dataset [csic_2010_preprocessed.parquet](./data/csic_2010_preprocessed.parquet)  
    - **Processing notebook:** [featurise.ipynb](./notebooks/featurise.ipynb)
    - **Output:** Featurised dataset [csic_2010_featurised.parquet](./data/csic_2010_featurised.parquet)  
3. **Training & evaluation**
    - **Input:** Featurized dataset [csic_2010_featurised.parquet](./data/csic_2010_featurised.parquet)    
    - **Processing notebook:** [evaluate.ipynb](./notebooks/evaluate.ipynb)
    - **Output:** [Trained baseline model](./models/model.joblib) & [metrics report](./report/metrics.json), [confusion matrix](./report/cm.png) **Fixme**

## Repository Structure

```
.
├── data/                       # Raw, preprocessed, and featurised datasets
│   ├── csic_2010_raw.csv
│   ├── csic_2010_preprocessed.parquet
│   └── csic_2010_featurised.parquet
├── models/                     # Trained models
│   └── model.joblib
├── notebooks/                  # processing stages
│   ├── preprocess.ipynb  # Exploratory data analysis
│   ├── featurise.ipynb         # Feature engineering
│   └── evaluate.ipynb          # Model training & evaluation
└── README.md
```

---

### 1. **Raw dataset Analysis:**    

- The raw dataset contains 223,585 parameter level rows and 18 columns.
- `index` column shows `36000` unique values whereas total rows count is `223585`, It implies that index column values are not unique. 
- `method` column with `3` unique values (GET, POST, PUT)
- `url` column with `1643` unique values. 
- `host` column has `2` unique values.
- `contentLength` column has `382` unique values. 
- `cookie` column has `61065` unique values. The `cookie` is the Unique Identifier for HTTP Requests 
- `payload` column with `33769` unique values. The `Payload` are split across multiple rows for the same request. We combine fragments to reconstruct the full payload.   
- `protocol`, `userAgent`, `pragma`, `cacheControl`, `accept`, `acceptEncoding`, `acceptCharset`, `acceptLanguage`, `connection`, `contentType` all these columns have a single unique value.
- The `label` is the target column with `anom` (anomalous) and `norm` (normal) values

Key Findings:  
1. Values of `index`, `method`, `url`, `host` and `label` columns are consistent (i.e same) for the same `cookie` value. That imples `cookie` is a Unique Identifier for HTTP Requests (one HTTP request)
3. `Payload` column values are fragmented and spread across multiple rows with same `cookie` value. All these Rows when grouped by `cookie` value share the same `label` classification
4. When the `223,585` rows are grouped by `cooke` they make up `61065` HTTP requests. 
5. Payloads for `23881` HTTP requests are fragmented across multiple rows. Each row 
in when grouped by `cookie` have a different fragment of the payload.  
6. `25491` HTTP requests have no payload. This is expected for GET requests
7. `contentLength` for `17,977` HTTP requests have non-null values whereas for 43,088 HTTP requests are `NaN/None` which is expected for GET requests 
8. The core metadata columns `cookie`, `index`, `method`, `url`, `host`, and `label`, `payload` are fully populated (`61,065` non-null entries). This ensures that each reconstructed HTTP request retains complete metadata. The grouping by `cookie` preserves the integrity of the original requests.  

Detailed analysis can be found here: [eda_and_analysis.ipynb](./notebooks/eda_and_analysis.ipynb)

The output is stored in [csic_2010_preprocessed.parquet](./data/csic_2010_preprocessed.parquet)  

---

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

Engineered features include:
- **Length-based**: `url_length`, `payload_length`, `contentLength`
- **URL patterns**: `url_has_numeric_id`, `url_has_select`, `url_has_drop`
- **Payload structure**: 'payload_quote_count', 'payload_paren_count' 'payload_slash_count', 'payload_parameter_count', 'payload_percent_count', 'payload_percent_encoded_count',
- **HTTP metadata**: one-hot encoded method_GET', 'method_POST', 'method_PUT', 'host_localhost:8080', 'host_localhost:9090'
- **Derived flags**: 'is_17_to_32_bytes', 'is_empty_payload',
- **Binary target**: 'anomalous'

Detailed analysis can be found here: [featurize.ipynb](./notebooks/featurize.ipynb)

The output is stored in [csic_2010_featurised.parquet](./data/csic_2010_featurised.parquet)  

---

### 3. **Training & evaluation**  
The training and target feature columns are identified  
The dataset is split into Training and Test data subsets in ratio `70:30`  
The baseline model is trained using `RandomForestClassifier`
The Feature Importances are found out from the model
The model is used to make predictions on the test data subset `30%`  
The model performance evaluated by comparing model predictions against the actual targets and deriving key metrics    
The model performance is compared against DummyClassifier and ConstantClassifier model predictions metrics. This guage the effectiveness of the baseline trained model
Cross Validation is used to further evaluate the model performance.


Detailed analysis can be found here: [featurize.ipynb](./notebooks/evaluate.ipynb)

Random Forest classifier basemodel the metric findings:
| Metric | Score |
| --- | --- |
| Accuracy | 0.8834 |
| Precision | 0.8981 |
| Recall | 0.8074 |
| F1-Score | 0.8504 |
| ROC-AUC | 0.9517 |
| Average Precision | 0.9464 |
| Class Imbalance Ratio | 1.44 |
| Number of Test Samples | 18320 |

- Accuracy `88.34%`, F1-Score: `85.04%`, and ROC-AUC: `95.17%` indicate the model performs well in classifying anomalies
- Precision `89.81%` and Average Precision: `94.64%`: High reliability in positive predictions (low false positives)
- Recall `80.74%`: Model captures most anomalies
- class_imbalance_ratio `1.44`: Indicates class imbalance i.e. ratio of norm/anom in the test targets
- `~19.26%` chance of false negatives. Further tuning hyperparameters, feature engineering, resampling may enhance recall. (Future Enhancement)

Findings from Feature importance analysis:
- Features with high imortance  (Importance >= 0.05): `payload_length(`, `url_length`, `payload_percent_count`, `payload_percent_encoded_count`, `contentLength`
- Features with moderate importance (0.01 <= Importance < 0.05): `url_has_numeric_id`, `payload_quote_count`, `payload_parameter_count`, `payload_slash_count`
- Features with low importance (Importance < 0.01): `method_PUT`, `payload_paren_count`, `host_localhost:9090`, `host_localhost:8080`, `is_17_to_32_bytes`, `method_GET`, `method_POST`, `is_empty_payload`, `url_has_select`, `url_has_drop`

---

## Future Work

- Advanced feature engineering
- Model comparison (XGBoost, gradient boosting, SVM, Logistic Regression)

---

**References:**  

- [CSIC 2010 HTTP Dataset in CSV Format by **Peter Scully**](https://petescully.co.uk/research/csic-2010-http-dataset-in-csv-format-for-weka-analysis/)  

---

*This is an academic experimental project exploring ML-based HTTP traffic anomaly detection.*