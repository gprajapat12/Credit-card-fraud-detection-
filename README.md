# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions.

## Dataset

Based on the [Credit Card Fraud Detection dataset (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- This project uses a sample of 9,965 transactions (38 fraud cases, about 0.38%)
- Features `V1`-`V28` are anonymized (PCA), plus `Time` and `Amount`
- Target column: `Class` (0 = normal, 1 = fraud)

The dataset file is not included in this repository. Download `creditcard.csv` from Kaggle and place it in the same folder as the notebook.

## Results

Best model: **Random Forest**, with the highest PR-AUC (0.972).

## Key Learnings

- Accuracy is misleading on imbalanced data: every model scores about 100% accuracy, so precision, recall and PR-AUC tell the real story.
- Class weights and SMOTE caught one more fraud (6 of 8 instead of 5) but produced a few false alarms, so there is a precision vs. recall trade-off.
- SMOTE must be applied to the training data only, never the test data.
- Random Forest ranked fraud cases best overall (highest PR-AUC).

## Limitations

- The test set contains only 8 fraud cases, so one extra catch changes recall a lot. The differences between models are small and should be treated with caution.
- Only one train/test split was used. Cross-validation and the full 284,807-row dataset would give more reliable results.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook credit_card_fraud_detection.ipynb
```

Or open the notebook in Google Colab and upload `creditcard.csv`.

## Tools Used

Python, pandas, scikit-learn, imbalanced-learn, matplotlib
