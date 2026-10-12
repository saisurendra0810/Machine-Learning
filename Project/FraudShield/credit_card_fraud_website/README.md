
# FraudShield AI — Credit Card Fraud Detection Website

A Python Flask website for the supplied `credit_card_fraud_2026.csv` dataset.

## Models
- Logistic Regression
- Support Vector Machine (SVM)
- Isolation Forest

The first two models are supervised classifiers. Isolation Forest is used as an additional anomaly signal in the ensemble prediction.

## Target
`is_fraud`

## Input features
24 features are used. `transaction_id` is excluded because it is only an identifier.

## Run on Windows

1. Install Python 3.10+.
2. Open Command Prompt in this folder.
3. Create a virtual environment:
   `python -m venv venv`
4. Activate it:
   `venv\Scripts\activate`
5. Install packages:
   `pip install -r requirements.txt`
6. Start:
   `python app.py`
7. Open:
   `http://127.0.0.1:5000`

The model trains automatically when `app.py` starts.
