## Overview

This script implements a **wrapper-style feature selection** for a **credit/default prediction** dataset using:

1. **SMOTE** to balance the classes
2. **StandardScaler** to normalize features
3. **Particle Swarm Optimization (PSO)** (via `pyswarms`) to search for good feature subsets
4. **KNN** as the classifier to evaluate each subset
5. **Information Value (IV)** as an additional relevance signal for selected features

The goal is to pick a subset of features that gives good classification performance while keeping the model compact.

> **Note:** The current PSO setup assumes a binary feature-selection setting but uses a continuous PSO backend; you may need to post-process particle positions (round to 0/1) or switch to a discrete/binary PSO variant if your environment supports it.

---

## Requirements

Install the following packages:

```bash
pip install pandas numpy scikit-learn imbalanced-learn pyswarms openpyxl
```

* `pandas` – read Excel file
* `scikit-learn` – preprocessing, KNN, metrics
* `imbalanced-learn` – SMOTE
* `pyswarms` – PSO optimizer
* `openpyxl` – Excel reader backend

---

## Data

The script expects an Excel file at:

```python
excel_file_path = '/content/drive/MyDrive/Colab-Notebooks/data.xlsx'
```

The file should contain:

* **features**: all columns except the target
* **target**: `Default_or_not` (0/1)

If your path or target column name is different, change it here:

```python
df = pd.read_excel(excel_file_path)
X = df.drop('Default_or_not', axis=1).values
y = df['Default_or_not'].values
```

---

## What the Script Does

1. **Load data** from Excel
2. **Split** into `X` (features) and `y` (label)
3. **Scale** features
4. **Apply SMOTE** to balance the dataset
5. **Run PSO** to search for a feature subset

   * For each particle (subset):

     * select columns
     * train KNN on train split
     * compute accuracy/F1
     * compute IV for the selected features
     * build a fitness value
6. **Train final KNN** on the best feature subset
7. **Print**:

   * selected feature indices
   * final Accuracy and F1-score

---

## Running

Just run the script:

```bash
python feature_selection_pso.py
```

(or run the cells in Colab if you’re using the same path structure).

---

## Important Notes

* **IV calculation** here is simplified: it bins each feature into 5 quantiles (`qcut`) and computes WOE/IV. For production credit-scoring, use domain-specific binning.
* **PSO part** uses `GlobalBestPSO` from `pyswarms`, which is continuous. To make it truly binary, round the positions to 0/1 inside the objective or use a discrete PSO variant.
* If your dataset is small, SMOTE + CV inside PSO can be slow — reduce `max_iter` or `n_particles`.

---

## Output Example

You should see logs like:

```text
Shape of X: (2044, 20), Shape of y: (2044,)
Shape of resampled X: (3632, 20), Shape of resampled y: (3632,)
Best subset of features: [ 0  3  7 12 15]
Final Accuracy: 0.91, Final F1 Score: 0.88
Selected Features: [ 0  3  7 12 15]
```

Adjust PSO settings for better results:

```python
num_particles=20
max_iter=200
```
