# Predictive maintenance system

## Live application
URL : https://predictivemaintenance-machines.streamlit.app/

Github Repo: https://github.com/Yashashwini1/predictive_maintenance.git


-----------------
Table of contents
-----------------

1. Problem Statement and system design
2. ML Pipeline and Model Evaluation results
3. Production readniness Critique
4. Web App deployment
----------------------------------------------------------------------------------

# 1. Problem Statement and system design


## Problem Statement: 
Unexpected machine failures can lead to production downtime, increased maintenance costs and operational disruptions. The objective of this project is to develop a predictive maintenance system capable of identifying machines at risk of failure before breakdown occurs. The system provides maintenance engineers with failure probability estimates and predicted failure types through a web application, enabling proactive maintenance planning and improved operational efficiency.
#### Dataset: 
The dataset is a synthetic predictive maintenance dataset that has failures that are encountered in the industry. It contains unique ID, productID,
features: air temerature, process temperature, roational speed, torque and tool wear taht represtenats the functioning of the machine. Target: Failure and type of failures occured in the machine.
#### Goal: 
A simple application for the maintenance engineers that detects an early failure warning of machines
#### User Interface:

The Streamlit application allows engineers to:

- Monitor fleet performance
- View machine failure probability
- Predict failure type
- Assess risk levels (Low / Medium / High)
- Receive recommended maintenance actions

#### Success:
The system is considered successful if it can:

- Detect failures early
- Reduce unplanned downtime
- Improve maintenance planning
- Minimize operational disruptions

#### Cost: In operational context,
##### False Negatives (Highest Cost)

A machine failure is missed by the model, resulting in:

- Unplanned downtime
- Equipment damage
- Emergency maintenance
- Supply chain disruption
- Increased repair costs
- overload on the backup machines
- delays
- backlog, sudden repair cost

#### False Positives

The model predicts a failure that does not occur, resulting in:

- Unnecessary maintenance activities
- Increased maintenance costs
- Resource allocation inefficiencies

#### Known Limitations

- The dataset contains 27 contradictory labels between `Target` and `Failure Type`.
- Random Failures were excluded from the failure-type model due to limited samples and label inconsistency.
- The binary model prioritizes recall over precision and may generate false alarms.

## System Design
![alt text](images/image.png)

Figure 1: The system consists of four layers:

### Data Layer
- data ingestion, 
- validation, 
- cleaning 
- preparation of machine sensor data 

### Preprocessing Layer
- feature engineering, 
- train-test splitting 
- preprocessing transformations. 
- The preprocessing pipeline is fitted only on training data and reused during inference to ensure consistency between training and production.

### Model Layer
Responsible for 
- model training, 
- evaluation, serialization, 
- deployment 
- serving. 
The model is evaluated using 
- precision, recall, F1-score and confusion matrix before being serialized for deployment.

Production enhancements such as monitoring, retraining, model registry and CI/CD are shown as future additions.

### End User Layer
Provides maintenance engineers with access to predictions through a Streamlit web application. Users can view failure probability, predicted failure type and risk level to support maintenance prioritization.

# Model Setection Stratergy:
This is a classification problem with class imbalance.

This is a classification problem with class imbalance.

## Model 1: Random Forest

**Purpose:** Failure Detection (Failure / No Failure)

Reasons for selection:

- Handles non-linear relationships effectively
- Robust to outliers
- Provides feature importance for explainability
- Strong balance between performance and interpretability

## Model 2: XGBoost

**Purpose:** Failure Type Classification

Reasons for selection:

- High predictive performance
- Works well with imbalanced datasets
- Excellent performance on structured data

> Note: Random Failures were excluded due to limited sample size(18).

-----------------------------------------------------------------------------------

# 2. ML Pipeline
-----------------------------------------------------------------------------------
The ml pipeline was designed to be modular, reproducible and production-oriented.
pipeline steps: ```text
Data Loading
      ↓
Data Validation
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Streamlit Deployment

Preprocessing is fitted exclusively on the training set and then applied to the test set to prevent data
leakage. To prevent data leakage, 
- the dataset was first split into training and test sets. 
- All preprocessing steps, including categorical encoding, were fitted exclusively on the training data. 
- The fitted preprocessing pipeline was then applied to the test data using the same learned transformations. This ensures that no information from the test set influences model training or feature engineering. 
- The trained model and preprocessing pipeline are serialized and stored for inference within the
Streamlit application.

 # Model Evaluation Results
 The dataset exhibits class imbalance because machine failures occur much less frequently than normal
operation. For this reason, accuracy alone is not an appropriate evaluation metric
Evaluation Metrics: Recall, Precision, F1 Score, Confusion Matrix
Why Recall?
- Recall was selected as the primary metric because the business objective is to detect as many true
failures as possible. 
- A missed failure (false negative) can result in: Unplanned downtime, Lost production, Expensive repairs
- Therefore, maximizing recall provides greater operational value than simply maximizing accuracy.

RandomForest: 
Failure Class (1)              

- Precision = 0.63
- Recall = 0.65
- F1 = 0.64
- Support = 68

Failure Class (0)              

- Precision = 0.99
- Recall = 0.99
- F1 = 0.99
- Support = 1932

The model struggles because of 1932 No Failure 68 Failure

XGBoost: The model is trained on only failures dataset i.e target is 1 and also removing random failures class is too small (Random Failures-18)
- Accuracy: 94%
- Heat Dissipation Failure: Precision 1.00 | Recall 1.00 | F1 1.00
- Overstrain Failure: Precision 0.92 | Recall 0.80 | F1 0.86
- Power Failure: Precision 0.87 | Recall 1.00 | F1 0.93
- Tool Wear Failure: Precision 0.90 | Recall 0.90 | F1 0.90

### Future Improvements

- Evaluate XGBoost for binary failure detection.
- Add SHAP explainability for individual predictions.
- Introduce automated drift monitoring.
- Implement CI/CD for automated deployment.
- Collect additional examples of Random Failures.
- Integrate maintenance history and machine operating conditions.

-----------------------------------------------------------------------------------
# 3.Production readniness Critique
----------------------------------------------------------------------------------
Monitoring Strategy:
- Data Drift: monitoring the incoming data and changes in features Air Temperature, Process Temperature, Rotational Speed,Torque, Tool Wear. checking for aditional features and that were added. 
- Model / prediction Drift: Model perfomance on failure prediction, for example : how many machines were predicted as failures a month ago and how many are we predicting currently. Retraining Based on the new ground truth information that was gathered from the engineers.
- Bussiness performaces: How many actual failures vs predictions. How many failures missed by the model. Unexpacted Maintenance activities and regular planned maintenance activities. Keeping the engineers and their knowledge in the loop

# Retraining stratergy:
- A scheduled retraining monthly or quaterly based on the data collection. 
- Retraining if any drifts in data or model is observed and more importantly retraining if there has been any maintenance activity on the machines. Changes to the data from the machines after maintenances or repairs.
- Retraining or re-running the pipeline if there are new machines on board and adjusting the metrics based on the new machines. Keeping the engineers and their knowledge in the loop. 

Before replacing the production model:
- Train candidate model
- Evaluate on the  current validation dataset
- Compare against current production model: Precision and recall 
- Deploy only if performance improves
- Maintain rollback capability

Present model: The model would be retrained periodically (e.g., monthly or quarterly) or when sufficient new machine data becomes available. 
In a production environment, 
- incoming sensor data and maintenance outcomes would be stored in a centralized data repository. 
- A scheduled retraining pipeline would extract the latest labeled data, perform the same preprocessing steps, 
    - retrain both the binary failure detection model and the failure-type classification model, and log all experiments, metrics, and model artifacts using MLflow.

In addition to scheduled retraining, retraining could also be triggered by 
- data drift or performance degradation. 
    - For example, if the distribution of torque, temperature, or rotational speed changes significantly from the training data, or if the observed failure detection rate drops below a predefined threshold, a retraining workflow would be initiated.

# Risks and how to mitigate them:
- Data quality risk: Production data may differ from the training data due to sensor errors, missing values, or values outside the expected range. This can be mitigated by adding schema validation, range checks, missing-value handling, and alerts when incoming data does not match expected distributions.

- Data Ingestion risk: If the ingestion pipeline is overloaded or delayed, predictions may not be available in time. This can be mitigated with queue-based ingestion, retry logic, logging, and monitoring of pipeline latency and failures.

- Class imbalance risk: Failure events are rare, and the distribution of failure types may change over time. For example, random failures may become more frequent in production. This can be mitigated by monitoring class distributions, retraining with updated data, using class weighting or resampling techniques, and reviewing rare failure categories separately.

- Model performance risk: The model may miss failures or generate too many false alarms. Since false negatives are more costly, recall and false negative rate should be monitored continuously. New models should only replace the live model after validation against the current production model.

- Data drift risk: Machine behavior may change due to new operating conditions, new equipment. This can be mitigated by monitoring feature drift for variables such as torque, rotational speed, temperature, and tool wear, and triggering retraining when drift exceeds a defined threshold.

- Prediction failure risk: The model service may fail or return invalid predictions. This can be mitigated with error handling, fallback rules, health checks, logging, and rollback to a previous stable model version.

- UI risk: Maintenance engineers may misinterpret the prediction or overtrust the model. The app should clearly show failure probability, risk level, confidence, and recommended action, along with known limitations. The system should be positioned as decision support, not a replacement for engineering



## Project structure

```text
predictive_maintenance_pipeline/
├── app.py
├── requirements.txt
├── data/
│   └── predictive_maintenance.csv
├── models/
├── artifacts/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train_binary_model.py
    ├── train_failure_type_model.py
    ├── train_pipeline.py
    ├── predict.py
    └── utils.py
```

## Important scripts

- `src/data_loader.py` loads the CSV, removes ID columns, and validates required columns.
- `src/preprocessing.py` creates the feature list, train/test split, and preprocessing transformer.
- `src/train_binary_model.py` trains the Random Forest binary failure model.
- `src/train_failure_type_model.py` trains the XGBoost failure type model.
- `src/train_pipeline.py` runs the full training pipeline and saves the models.
- `src/predict.py` contains reusable hierarchical prediction logic.
- `app.py` is the Streamlit web app.

## How to run locally

Github Repo: https://github.com/Yashashwini1/predictive_maintenance.git

Install dependencies:

```bash
pip install -r requirements.txt
```

dataset:

```text
data/predictive_maintenance.csv
```

Train the models:

```bash
python src/train_pipeline.py
```

Run the app:

```bash
streamlit run app.py
```


