 Supervised Machine Learning Models

Regression and classification with Scikit-learn on the cleaned loan applicant dataset from Week 1
(Python for Machine Learning and Data Preprocessing).

Five algorithms are implemented and compared: **Linear Regression, Logistic Regression, Decision Tree,
Random Forest and K-Nearest Neighbors**.

| Problem | Target | Models compared |
|---|---|---|
| Regression | `Annual_Income` | Linear Regression, KNN, Decision Tree, Random Forest |
| Classification | `Approved` (0 / 1) | Logistic Regression, KNN, Decision Tree, Random Forest |

## Results

Evaluation uses an 80/20 train/test split (412 / 103 rows), 5-fold cross-validation and
GridSearchCV for tuning. The best model is chosen by cross-validation, not by the test score.

**Regression** (income is a z-score; errors converted to rupees)

| Model | MAE | RMSE | R2 (test) | R2 (cross-val.) | Avg. error (Rs.) |
|---|---|---|---|---|---|
| Linear Regression | 0.328 | 0.473 | 0.760 | 0.701 | 74,400 |
| Random Forest | 0.312 | 0.441 | 0.791 | 0.688 | 70,812 |
| KNN Regressor | 0.463 | 0.568 | 0.654 | 0.598 | 105,032 |
| Decision Tree | 0.377 | 0.504 | 0.727 | 0.547 | 85,509 |

**Classification** (tuned models, test set)

| Model | Best parameters | CV AUC | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | C=0.1 | 0.738 | 0.718 | 0.785 | 0.756 |
| Random Forest | max_depth=4, min_samples_leaf=3, n_estimators=200 | 0.721 | 0.709 | 0.786 | 0.752 |
| Decision Tree | max_depth=2, min_samples_leaf=1 | 0.713 | 0.709 | 0.758 | 0.712 |
| KNN | n_neighbors=21 | 0.686 | 0.670 | 0.764 | 0.751 |

**Findings**
- Linear Regression is the best regressor and Logistic Regression the best classifier. The simple linear models
  match the tree-based ones, so the simpler model is the better choice.
- Credit score is by far the most important feature for loan approval.
- Tuning helped most for the Decision Tree and KNN.
- The dataset is small (515 rows) and synthetic, so small differences between models are not reliable.

Full explanation, code, outputs and charts: [`report/Week2_Supervised_ML_Models_Report.pdf`](report/Week2_Supervised_ML_Models_Report.pdf)

## Repository structure

```
data/loan_applicants_clean.csv   cleaned dataset from Week 1
src/train_models.py              trains, tunes and evaluates all models
src/build_report.py              builds the PDF report
results/                         metric tables (CSV), charts (PNG), log
models/                          saved best models (joblib)
report/                          PDF report
```

## How to run

```bash
pip install -r requirements.txt
python3 src/train_models.py      # trains models, writes results/ and models/
python3 src/build_report.py      # optional: rebuilds the PDF report
```

## Using the saved models

```python
import joblib
clf = joblib.load("models/best_classifier.joblib")   # Logistic Regression
reg = joblib.load("models/best_regressor.joblib")    # Linear Regression
```

The models expect inputs preprocessed exactly like `data/loan_applicants_clean.csv` (encoded and standardised,
same column order as in `src/train_models.py`).
