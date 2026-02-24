# Recruitment Predictor

This project builds a machine learning pipeline to predict `HiringDecision` from the Kaggle-style recruitment dataset.

## Project Files
- `recruitment_working.ipynb`: Main end-to-end notebook.
- `recruitment_data.csv`: Input dataset.
- `recruitment.ipynb`: Earlier notebook version.
- `test.ipynb`: Notebook rendering test file.
- `artifacts/best_recruitment_model.joblib`: Exported best model.
- `artifacts/model_summary.json`: Saved model selection summary.

## Workflow Overview
The notebook is organized to run top-to-bottom:

1. **Step 0: Setup + Load Data**
- Import libraries.
- Load dataset.
- Inspect shape, dtypes, and missing values.

2. **Step 1: Data Preparation**
- Handle missing values (numeric median, categorical mode).
- Encode categorical fields when needed (`Gender`, `EducationLevel`).
- Confirm/create target label (`HiringDecision`).
- Engineer derived features:
  - `ExperienceBucket`, `ExperienceBucketCode`
  - `TechnicalStageScore`
  - `BehavioralStageScore`
  - `OverallCompositeScore`

3. **Step 2: EDA + Hypothesis Testing**
- Class distribution.
- Feature-by-target boxplots.
- Correlation heatmap.
- Statistical tests:
  - t-test (`InterviewScore` by `HiringDecision`)
  - chi-square (`EducationLevel` vs `HiringDecision`)
  - chi-square (`ExperienceBucket` vs `HiringDecision`)

4. **Step 3: ML Pipeline**
- Train/test split with stratification.
- Preprocessing pipelines with `ColumnTransformer`.
- Models trained/tuned:
  - Logistic Regression (baseline)
  - Logistic Regression (L1/L2 regularized, GridSearchCV)
  - Decision Tree (tuned + pruning)
  - KNN (tuned)
- Bias-variance diagnostics:
  - Validation curve (tree depth)
  - Learning curve (logistic)

5. **Step 4: Multicollinearity (VIF)**
- Compute VIF for numeric features.
- Iteratively remove high-VIF features.
- Refit tuned logistic model on VIF-reduced features.
- Compare before vs after.

6. **Step 5: Class Imbalance Handling**
- Evaluate class ratio.
- Train class-weighted logistic and tree models.
- Optional SMOTE pipeline (if `imblearn` is installed).
- Compare with baseline models.

7. **Step 6: Finalization**
- Select best candidate by **F1-score**.
- Save best model and summary to `artifacts/`.
- Plot feature importance (tree) or top coefficients (linear).

## Evaluation Metric
Primary metric: **F1-score**.

Why:
- Balances precision and recall.
- Better than raw accuracy when class distribution is uneven or when both false positives and false negatives matter in hiring decisions.

## Current Best Model
From the results shared during development, the top performer was:
- **Decision Tree (tuned + pruning)** with strongest F1/ROC-AUC among tested models.

## How to Run
1. Open `recruitment_working.ipynb`.
2. Run all cells in order from Step 0 to Step 6.
3. Review model comparison tables.
4. Use the exported model in `artifacts/`.

## Requirements
Typical Python packages used:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`
- `scikit-learn`
- `statsmodels`
- optional: `imbalanced-learn` (for SMOTE)
