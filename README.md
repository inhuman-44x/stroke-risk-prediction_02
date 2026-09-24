# Stroke Risk Prediction: Model Comparison

A machine learning project comparing **Logistic Regression, Random Forest, and XGBoost** for predicting stroke occurrence from demographic and clinical characteristics.

The project evaluates model performance using  **ROC-AUC, PR-AUC, calibration, and risk stratification** to assess how well predicted probabilities distinguish individuals with and without stroke and separate the test population into groups with different observed stroke rates.

## Project Overview

Stroke is a major global health problem, and identifying individuals at elevated risk can support preventive strategies and more targeted clinical assessment.

* **Logistic Regression**
* **Random Forest**
* **XGBoost**

Rather than evaluating models using accuracy alone, the analysis focuses on metrics appropriate for a substantially imbalanced outcome and on the quality of predicted probabilities.

The analysis addresses three primary questions:

1. How well does each model discriminate between individuals with and without stroke?
2. How well do predicted probabilities correspond to observed stroke frequency?
3. Can predicted probabilities separate the test population into groups with meaningfully different observed stroke rates?

## Dataset

The project uses the **Stroke Prediction Dataset** originally available on Kaggle:

(https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

The dataset contains **5,110 observations and 12 variables**, including demographic and clinical characteristics such as:

* Age
* Gender
* Hypertension
* Heart disease
* Marital status
* Work type
* Residence type
* Average glucose level
* BMI
* Smoking status

The target variable is `stroke`, indicating whether an individual experienced a stroke.

Stroke is relatively uncommon in the dataset, with approximately **4.9% of observations classified as stroke cases**. This class imbalance is an important consideration when evaluating model performance.

## Modelling

### Logistic Regression

Logistic Regression was used as an interpretable baseline model. Random Forest was used to evaluate whether a nonlinear tree-based ensemble could improve predictive performance XGBoost was included as a gradient-boosting approach capable of modelling nonlinear relationships and interactions.

For all three models, **PR-AUC (average precision)** was used as the primary hyperparameter-tuning metric because of the substantial class imbalance.

## Model Evaluation

Models were evaluated on the held-out test set using:

### ROC-AUC

Measures the model's ability to distinguish between individuals with and without stroke across classification thresholds.

### PR-AUC

Measures the precision-recall trade-off and is particularly informative for imbalanced outcomes such as stroke in this dataset.

### Brier Score

Measures the accuracy of predicted probabilities, with lower values indicating better probabilistic accuracy.

### Calibration

Calibration curves were examined to assess agreement between predicted probabilities and observed stroke frequencies.

### Risk Stratification

The selected model's predicted probabilities were used to divide the test population into Low, Moderate, and High-risk strata. Observed stroke rates were then compared across these groups.

These strata are **model-derived and have not been externally validated**. They therefore represent risk separation within this dataset rather than established clinical risk categories.

## Model Performance

### Cross-Validation

The best hyperparameter configuration for each model was identified using 5-fold stratified cross-validation, with PR-AUC as the optimisation metric.

### Held-Out Test Set

The final models were evaluated on the previously unseen test set.

XGBoost achieved the highest test-set PR-AUC (**0.240**) and was therefore carried forward for threshold optimisation, risk stratification, and feature interpretation.

Bootstrap resampling was additionally used to estimate **95% confidence intervals for ROC-AUC and PR-AUC** on the held-out test set.

### Follow-up on the same split

`src/train_improved.py` and `src/train_blend.py` reuse the processed train and test tables. Models are chosen by cross-validated PR-AUC. The held-out logistic regression baseline matches the notebook (**ROC-AUC 0.837**, **PR-AUC 0.219**).

A random forest with age interactions and a high-glucose flag, followed by isotonic calibration, is the best follow-up result:

| Model | ROC-AUC | PR-AUC | Brier score |
| ----- | ------: | -----: | ----------: |
| Notebook logistic regression | 0.837 | 0.219 | 0.0418 |
| Notebook XGBoost | 0.827 | 0.240 | 0.0418 |
| Calibrated random forest | 0.828 | 0.246 | 0.0414 |

PR-AUC rises from **0.240** to **0.246**, and the Brier score falls from **0.0418** to **0.0414**. ROC-AUC does not improve on **0.837**. Class-weighted XGBoost was not retained: its Brier score rose to **0.171**. Saved metrics are in `outputs/improved_results.json` and `outputs/blend_results.json`.

## Threshold Optimisation

Because the objective of risk stratification is to identify individuals who may be at elevated risk, classification thresholds were selected using out-of-fold training predictions rather than directly optimising thresholds on the held-out test set.

The primary threshold was selected using the following criterion:

* Minimum recall of **0.80**
* Among thresholds meeting this requirement, select the threshold with the highest precision.

A second threshold was selected to define the High-risk group:

* Minimum precision of **0.30**
* Among thresholds meeting this requirement, select the threshold with the highest recall.

The resulting thresholds were then applied unchanged to the held-out test set.

## Risk Stratification Results

The selected XGBoost model demonstrated clear separation in observed stroke rates across the three prediction-based strata.

| Risk stratum  |   n | Stroke events | Mean predicted risk | Observed stroke rate | Test population |
| ------------- | --: | ------------: | ------------------: | -------------------: | --------------: |
| Low risk      | 715 |            10 |                1.0% |                 1.4% |           70.0% |
| Moderate risk | 297 |            37 |               12.5% |                12.5% |           29.1% |
| High risk     |  10 |             3 |               38.7% |                30.0% |            1.0% |

Observed stroke rates increased progressively across the strata:

**1.4% → 12.5% → 30.0%**

## Limitations

### Dataset limitations

The analysis uses a single publicly available dataset, limiting the extent to which the findings can be generalized to other populations, demographic groups, and healthcare settings. The dataset is also cross-sectional, meaning that predictors and stroke status are observed at a single point in time. As a result, the models estimate the association between patient characteristics and the presence of stroke within the dataset, but cannot capture how individual risk changes over time or distinguish between factors that precede stroke and those that may change as a consequence of disease.

In addition, the dataset contains a relatively limited set of clinical and demographic variables and does not include continuously measured physiological or behavioral information. This limits the ability of the models to capture dynamic changes in risk that may occur between clinical assessments.

### Class imbalance

Only approximately 4.9% of observations experienced stroke. This limits the amount of positive-class information available for model training and contributes to the relatively modest PR-AUC

### Future work
Future work will therefore extend this approach of this work using **longitudinal datasets**, which can support modelling of changes in risk and prediction of future stroke events over defined time horizons.