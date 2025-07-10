# Random Forest: Intrusion Detection on the KDD Dataset

A supervised machine learning project using a Random Forest classifier to detect and classify network intrusions based on the KDD99 dataset. This model includes data preprocessing, hyperparameter tuning, performance evaluation, and visualization.

---

## Reproducibility Note

While repeated runs of this code generally produce **consistent results**, minor variation may occur due to random processes during training and environment-specific factors.  
⏱️ *Please allow up to 10 minutes for the script to complete.*

---

## Requirements

Ensure Python 3.7+ is installed.

**Install with pip:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**Or with conda:**
```bash
conda install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## Dataset Files

Ensure the following files are located in the **same directory** as the Python script (`Random_Forest.py`):

- `KDDTrain.txt`
- `KDDTest.txt`

These files are preprocessed KDD99 datasets.

---

## How to Run

Execute the script from your terminal or command prompt:

```bash
python Random_Forest.py
```

The following will occur:
1. Dataset loading and cleaning
2. Categorical feature encoding
3. Data standardization
4. Model training with hyperparameter tuning (`RandomizedSearchCV`)
5. Evaluation & reporting
6. Visualization of results

---

## Output & Interpretation

### Performance Metrics:
- **Classification Report**: Per-class precision, recall, and F1-score
- **Accuracy**: Total correct predictions
- **ROC-AUC (OvR)**: One-vs-Rest curves to show class separability

### Visualizations:
- **Confusion Matrix**: Highlights true vs. predicted class distribution
- **ROC Curves**: Measures discrimination ability between all 5 classes
- **Feature Importance Plot**: Identifies most influential features

*Close each plot window to proceed to the next.*

---

## Key Implementation Notes

- Label encoding is applied to categorical features: `protocol_type`, `service`, `flag`
- StandardScaler is used to normalize numerical features
- The “other” attack type is dropped to prevent issues during label encoding
- RandomizedSearchCV with 3-fold CV optimizes parameters like:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `bootstrap`

---

## Contact & Contributions

- Ensure data paths are correct if running on a different machine.
- For feedback, suggestions, or contributions, please open an issue or pull request.

---