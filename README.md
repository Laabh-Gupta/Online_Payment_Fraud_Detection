# Online Payment Fraud Detection

**Exploratory classification of synthetic payment transactions.**

Python notebooks for transaction analysis, feature preparation and comparisons involving Logistic Regression, Random Forest and XGBoost. This is an offline learning/research project; it does not expose a real-time fraud-detection API or verified live demo.

## Data and method

The original project uses the [Kaggle online payment fraud dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection). Inputs include transaction type/amount and origin/destination balances; `isFraud` is the target.

[model.ipynb](model.ipynb) contains preprocessing, visualization and model experiments. [model2.ipynb](model2.ipynb) contains additional work. Inspect the notebook's loading cells and update local dataset paths before execution.

## Explore locally

```bash
git clone https://github.com/Laabh-Gupta/Online_Payment_Fraud_Detection.git
cd Online_Payment_Fraud_Detection
python -m venv .venv
```

Activate the environment, then install notebook dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyterlab
jupyter lab
```

There is no locked environment; API changes may require a compatible historical library version or notebook adjustments.

## Evaluation boundaries

The notebooks explore accuracy, precision, recall, F1 and confusion matrices. No new or independently reproduced benchmark is claimed. Fraud is imbalanced, so accuracy alone is inadequate.

Before interpreting the current outputs: the Random Forest print cell refers to another model's recall; the XGBoost experiment uses its test partition for early stopping and computes AUC from class labels. A reproducible comparison needs separate validation/test partitions and probability-based AUC. These issues are documented rather than presenting the existing numbers as deployment evidence.

**Python · pandas · scikit-learn · XGBoost · Jupyter**<br>
[License](LICENCE).
