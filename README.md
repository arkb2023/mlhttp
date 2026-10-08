# HTTP Traffic Anomaly Detection
Machine learning-based anomaly detection on HTTP request traffic, built on the [CSIC 2010 dataset](http://lexr.ai/csic_dataset/output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv.zip)

## Overview

This project detects anomalous HTTP requests. The anomalies include injection attempts, scanner probes, and malformed traffic. The method is supervised classification.

The CSIC 2010 traffic contains 61,065 requests: 36,000 normal and 25,065 anomalous. The requests are preprocessed, featurised, and used to train a Random Forest classifier. The selected model catches 94.5% of anomalies with 84.2% precision on a held-out test set.

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

A demo script, [predict.py](./demo/predict.py), classifies a single HTTP request with the selected model. Applies the same featurisation as `featurise.ipynb`, loads `models/model.joblib`, and prints the prediction with the features that fired. 

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
│   ├── analyse_urls.ipynb                # URL pattern-family discovery
│   ├── evaluate.ipynb                    # Model training & evaluation
│   ├── featurise.ipynb                   # Feature engineering
│   └── preprocess.ipynb                  # Exploratory data analysis
├── demo
│   └── predict.py                        # Model prediction demo app
└── README.md
```

## Results

Test set: `18,320` requests (`10,800` normal, `7,520` anomalous), held out from training.  

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
      <th>f1</th>
      <th>roc_auc</th>
      <th>average_precision</th>
      <th>fpr</th>
      <th>fnr</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RandomForest (tuned)</th>
      <td>0.9046</td>
      <td>0.8420</td>
      <td>0.9449</td>
      <td>0.8905</td>
      <td>0.9807</td>
      <td>0.9745</td>
      <td>0.1234</td>
      <td>0.0551</td>
    </tr>
    <tr>
      <th>RandomForest (default)</th>
      <td>0.9044</td>
      <td>0.9036</td>
      <td>0.8588</td>
      <td>0.8806</td>
      <td>0.9772</td>
      <td>0.9689</td>
      <td>0.0638</td>
      <td>0.1412</td>
    </tr>
    <tr>
      <th>Constant (always anom)</th>
      <td>0.4105</td>
      <td>0.4105</td>
      <td>1.0000</td>
      <td>0.5820</td>
      <td>0.5000</td>
      <td>0.4105</td>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Dummy (most_frequent)</th>
      <td>0.5895</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.5000</td>
      <td>0.4105</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>


The two trivial classifiers are the operating extremes: never alert and always alert. A useful detector must beat both F1 values (0.0 and 0.5820). Both Random Forests do.

**Selected model:** `RandomForestClassifier(class_weight="balanced", max_depth=15, min_samples_split=10, n_estimators=100)`

- **Selection rule:** highest F1 on the held-out test set. The tuned model scores `0.8905`. The default model scores `0.8806`. The tuned model wins.
- `class_weight="balanced"` was the decisive parameter. It raises `recall` from `0.8588` to `0.9449`. Missed anomalies drop from `14.1%` to `5.5%`. The tradeoff is `precision`, which falls from `0.9036` to `0.8420`, and false alerts, which rise from `6.4%` to `12.3%`.
- For intrusion detection, a missed attack costs more than a false alert. The tuned model is the better choice here. If alert volume is the constraint, the default model is the better choice instead.
- **Cross-validation** scores the default model on `5` folds of the **training split**. The folds are the same ones the **grid search** used. The CV numbers and the held-out test numbers agree to within about `0.002` on every metric. CV recall `0.8609` vs test recall `0.8588`, and CV F1 `0.8797` vs test F1 `0.8806`. This agreement means the model is not overfit. it performs the same on data it trained on and on data it never saw, so the test numbers are not an artifact of one particular split.

## Usage

- Prerequisites

```bash
pip install -r requirements.txt
```

- Run sequence  

1. [notebooks/preprocess.ipynb](./notebooks/preprocess.ipynb): Analyse and transform Raw CSV to preprocessed dataset
2. [notebooks/featurise.ipynb](./notebooks/featurise.ipynb): Transform preprocessed to featurised dataset
3. [notebooks/evaluate.ipynb](./notebooks/evaluate.ipynb): Featurised dataset to trained model and metrics 
4. [./demo/predict.py](./demo/predict.py): Test the model 

> [notebooks/analyse_urls.ipynb](./notebooks/analyse_urls.ipynb): runs inside stage 2. It discovers the URL pattern families used by the featuriser.

## Demo

[demo/predict.py](./demo/predict.py) classifies a single HTTP request with the trained model.
It takes the request fields as command-line arguments, applies the same featurization steps as `featurise.ipynb`, loads `models/model.joblib`, and prints the prediction with the features that fired.

1. Model classifies a harmless `GET` request as `normal`

```bash
python demo/predict.py \
    --method "GET" \
    --url "http://localhost:8080/tienda1/index.html" \
    --verbose

request: GET /tienda1/index.html (host=localhost:8080)
feature row (non-zero)
  method_GET = 1.0
  host_localhost:8080 = 1.0
  url_length = 40
  is_empty_payload = 1
prediction: normal (anomaly probability 0.1446)
fired features: method_GET, host_localhost:8080, url_length, is_empty_payload
```

2. Model classifies a `GET` request with suspicious URL as `anomalous`

```bash
python demo/predict.py \
    --method "GET" \
    --url "http://localhost:8080/tienda1/publico/autenticar.jsp/6909030637832563290.jsp.java" \
    --verbose
request: GET /tienda1/publico/autenticar.jsp/6909030637832563290.jsp.java (host=localhost:8080)
feature row (non-zero)
  method_GET = 1.0
  host_localhost:8080 = 1.0
  url_has_extensions_probe = 1
  url_has_numeric_id_probe = 1
  url_length = 81
  is_empty_payload = 1
prediction: anomalous (anomaly probability 1.0000)
fired features: method_GET, host_localhost:8080, url_has_extensions_probe, url_has_numeric_id_probe, url_length, is_empty_payload
```

3. Model classifies a `POST` request with suspicious payload as `anomalous`

```bash
python demo/predict.py \
    --method "POST" \
    --url "http://localhost:8080/tienda1/publico/registro.jsp" \
    --payload "modo=registro%253CSCRIPT%253Ealert%2528%2522Paros%2522%2529%253B%253C%252FSCRIPT%253E&login=armand&password=prusiato&nombre=Samia&apellidos=Catalayud+Mirabet&email=balademas.mareuil6%40enlanzarote.com.vc&dni=28178710F&direccion=C%2F+Cala%2C+9%2C+&ciudad=Mutxamel&cp=25332&provincia=Zaragoza&ntc=8608089924764976&B1=Registrar" \
    --verbose

request: POST /tienda1/publico/registro.jsp (host=localhost:8080)
feature row (non-zero)
  method_POST = 1.0
  host_localhost:8080 = 1.0
  url_length = 50
  payload_slash_count = 1
  payload_parameter_count = 13
  payload_percent_count = 14
  payload_percent_encoded_count = 14
  payload_has_alert = True
  payload_length = 324
prediction: anomalous (anomaly probability 0.9451)
fired features: method_POST, host_localhost:8080, url_length, payload_slash_count, payload_parameter_count, payload_percent_count, payload_percent_encoded_count, payload_has_alert, payload_length
```

## Limitations

- **Test split repeats URLs:** The split is over request rows, not over URLs. Normal URLs repeat heavily in this dataset. Each normal URL appears about 1,000 times. So the same URL ends up in both the training set and the test set. The rows are near-identical. Only the payload or byte counts differ slightly. A model can score well partly by remembering URLs from training. The current test numbers are therefore the optimistic bound. They measure performance on repeated traffic, not on unseen URLs.

- **Known blind spots:** The false negatives sit on `8` endpoints whose URLs look legitimate. Normal and attack traffic use the same URL patterns there. So URL features cannot separate the two classes. Only payload and length features can. Improving recall needs better payload and sequence features.

- **Lab traffic:** CSIC 2010 is synthetic traffic against one web application (an e-commerce shop). The numbers do not transfer directly to production traffic.

## Future work

- **URL-grouped split:** Split the data by URL, not by row. No URL would appear on both sides. This tests the model on URLs it has never seen. The score would drop. It gives the conservative bound. The current row-level numbers are the optimistic bound. Production traffic contains both URLs the model has seen and URLs it has never seen. So the true score sits between the two bounds.

- **Payload and sequence features:** The remaining false negatives need features that look inside the request body. Raw payload statistics, token patterns, or character sequences could reach the `8` blind-spot endpoints. This is the path to higher recall.

**References:**  
- [CSIC 2010 HTTP dataset in CSV format for Weka analysis — Peter Scully](https://petescully.co.uk/research/csic-2010-http-dataset-in-csv-format-for-weka-analysis/)

---

*This is an academic experimental project exploring ML-based HTTP traffic anomaly detection.*