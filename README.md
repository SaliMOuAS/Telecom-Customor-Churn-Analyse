# Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

End-to-end churn analysis for a telecom provider: from data cleaning and exploratory analysis to model comparison, churn drivers and customer risk segments with concrete retention recommendations.

## Business question

Which customers are likely to cancel, what drives their decision, and where should a retention team act first?

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data): **7,043 customers**, 21 columns (contract, tenure, services, billing, payment method) and a binary `Churn` target. **26.5 %** of the customers churned.

## Approach

1. **Cleaning:** `TotalCharges` converted to numeric, ID column removed, target mapped to 0/1
2. **Exploration:** churn rate by feature, numeric features compared between churners and non-churners
3. **Preprocessing pipeline:** median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features (scikit-learn `ColumnTransformer`)
4. **Models:** Logistic Regression and Random Forest, both with class weighting to handle the class imbalance
5. **Evaluation:** stratified 80/20 train/test split, 5-fold cross-validation, ROC-AUC, PR-AUC, accuracy, F1, precision and recall
6. **Interpretation:** Logistic Regression coefficients and Random Forest feature importances
7. **Business view:** customers grouped into five risk bands with recommended actions per band

## Results

Hold-out test set (1,409 customers, `random_state=42`):

| Model | ROC-AUC | PR-AUC | Accuracy | F1 (churn) | Recall (churn) | Precision (churn) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.633 | 0.738 | 0.614 | 0.783 | 0.504 |
| Random Forest | 0.843 | 0.653 | 0.759 | 0.630 | 0.775 | 0.531 |

5-fold cross-validation ROC-AUC on the training data: 0.845 (Logistic Regression) and 0.849 (Random Forest).

Both models find about **78 % of the churners**. Because the classes are imbalanced, accuracy is not the main metric here. Recall and PR-AUC matter more for a retention use case.

### What drives churn

| Increases churn risk | Reduces churn risk |
|---|---|
| Month-to-month contract | Long tenure |
| Short tenure | Two-year contract |
| Fiber optic internet | One-year contract |
| No online security, no tech support | |
| Payment by electronic check | |

The month-to-month contract and tenure are the two strongest signals in the Random Forest. The Logistic Regression ranks tenure and the two-year contract highest.

### Risk segments (Random Forest)

| Predicted risk band | Customers | Actual churn rate |
|---|---|---|
| Very low (0–20 %) | 473 | 3.2 % |
| Low (20–40 %) | 295 | 16.3 % |
| Medium (40–60 %) | 215 | 30.7 % |
| High (60–80 %) | 274 | 47.4 % |
| Very high (80–100 %) | 152 | 75.7 % |

The risk bands separate customers well: in the highest band three out of four customers actually churned, in the lowest band only about one in thirty.

### Recommendations derived from the analysis

- **High risk:** proactive retention calls, loyalty discounts, priority support
- **Medium risk:** satisfaction surveys, targeted offers, usage tips
- **Low risk:** regular communication, cross-selling, referral and loyalty programs
- **Structural levers:** move month-to-month customers to longer contracts and bundle security and support services

## Project structure

```
telecom_chun_end_to_end.py        # main pipeline: EDA, models, drivers, risk bands, exports
Analyse.py                        # extended analysis report with business context
WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset
requirements.txt
```

Running the main script writes its results (coefficients, feature importances, risk bands, predictions, summary) to an `outputs/` folder.

## Run it yourself

```bash
pip install -r requirements.txt
python telecom_chun_end_to_end.py
```

The script reads the CSV from the same folder, so no path needs to be changed.

## Notes

- The numbers above come from running `telecom_chun_end_to_end.py` on the included dataset. Small deviations are possible with other library versions.
- The two CSV files in the repository contain the same data, only the line endings differ.
