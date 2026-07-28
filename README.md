# Travel Insurance Claim Prediction

Course project for **CSE422: Artificial Intelligence (AI)**.

This project builds and compares three supervised learning models, a Decision Tree, a Logistic Regression classifier, and a feedforward Neural Network, to predict whether a travel insurance policy will result in a claim. The target label is heavily imbalanced (roughly 1.5% of policies result in a claim), so the project focuses as much on handling that imbalance correctly as on the models themselves.

## Problem Statement

Given details of a travel insurance policy (agency, product, destination, duration, sales amount, commission, and customer age), predict whether that policy will result in a claim (`Yes` / `No`). This is a binary classification task that can help insurers flag high-risk policies for pricing and risk management.

## Dataset

- **Source:** Travel Insurance dataset (Given by the course instructor), included in this repo as `travel insurance.csv`.
- **Size:** 63,326 records, 10 input features plus the target column `Claim`.
- **Features:** Agency, Agency Type, Distribution Channel, Product Name, Destination, Gender, Duration, Net Sales, Commission (in value), Age.
- **Class balance:** ~1.5% `Yes`, ~98.5% `No`.

### Data quality issues handled

- `Gender` is missing on about 71% of records, so the column is dropped rather than dropping the rows.
- A small number of records have a negative `Duration`, which is not physically possible and is removed.
- `Age` contains a placeholder value of 118 on 984 records; these are treated as missing/invalid and removed.
- `Net Sales` has negative values corresponding to refunds/cancellations; these are kept and converted into a binary `Is_Refund` flag rather than dropped.

## Methodology

1. **Cleaning:** drop the `Gender` column, remove invalid `Duration`/`Age` records, add an `Is_Refund` flag.
2. **Train/test split:** 70/30 stratified split on `Claim`, performed **before** encoding or scaling to avoid any leakage of test-set categories into preprocessing.
3. **Encoding & scaling:** a `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder(handle_unknown='ignore')` to categorical features, fit only on the training set.
4. **Modeling:**
   - **Decision Tree** — `class_weight='balanced'`, regularized with `max_depth=10` and `min_samples_leaf=20` to prevent overfitting.
   - **Logistic Regression** — `class_weight='balanced'`.
   - **Neural Network** (Keras) — 3 hidden layers (64/32/16 units), class-weighted, trained with early stopping on validation AUC.
5. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix, and ROC/AUC for each model. AUC and recall on the minority class are treated as the primary metrics, since accuracy is misleading on an imbalanced target.

## Results

| Model | Accuracy | Recall (Yes) | AUC |
|---|---|---|---|
| Decision Tree | 0.715 | 0.71 | 0.735 |
| Logistic Regression | 0.797 | 0.71 | 0.818 |
| Neural Network | 0.776 | 0.74 | 0.824 |

All three models catch roughly 71 to 74% of actual claims. The Neural Network has the highest AUC, followed closely by Logistic Regression, with the Decision Tree trailing but still clearly better than random guessing (AUC 0.5).

## Repository Structure

```
.
├── Travel Insurance Claim Prediction.ipynb   # Full notebook: EDA, preprocessing, all 3 models, plots
├── travel insurance analysis.py              # Same pipeline as a standalone script
├── travel insurance.csv                      # Dataset
├── Project Report.docx                       # Written report (methodology, results, discussion)
└── README.md
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn

Install with:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

## Usage

Run the full pipeline as a script:

```bash
python travel insurance analysis.py
```

Or open the notebook for a step-by-step walkthrough with inline plots:

```bash
jupyter notebook Travel Insurance Claim Prediction.ipynb
```

Both expect `travel insurance.csv` to be in the same directory.

## Key Takeaways

- Accuracy is not a reliable metric on an imbalanced target; a model that always predicts `No` already scores ~98.5% accuracy while being useless.
- Applying class-imbalance handling (`class_weight='balanced'`) consistently across every model, rather than just one, is what allows Logistic Regression and the Neural Network to actually detect claims instead of defaulting to the majority class.
- An unconstrained Decision Tree overfits severely on this dataset; regularizing depth and leaf size is necessary for it to generalize.

## Author

Project submitted for CSE422: Artificial Intelligence (AI).
