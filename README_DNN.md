# Deep Neural Network: Intrusion Detection on the NSL-KDD Dataset

A supervised learning project using a Deep Neural Network (DNN) to detect binary network intrusions based on the NSL-KDD dataset. This implementation includes data preprocessing, encoding, normalization, model training, evaluation, visualization, and model export.

---

## Reproducibility Note

Model results are generally **stable across runs**, though slight variations may occur due to random initial weights and system-specific behavior.  
*Allow up to 5–10 minutes to complete model training and evaluation in Colab.*

---

## Requirements

This script is intended to run in [Google Colab](https://colab.research.google.com) (recommended for simplicity). All required packages are pre-installed.

If running locally, ensure the following libraries are installed:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

---

## Dataset Files

Ensure the following files are available **before execution**:

- `KDDTrain.txt`
- `KDDTest.txt`

> The files must be **exactly named** and uploaded manually in Colab during execution.

---

## How to Run (via Google Colab)

1. Go to [https://colab.research.google.com](https://colab.research.google.com)  
2. Click **"New Notebook"**  
3. Copy the entire content of your `DNN.py` script and **paste it into the first code cell**  
4. Run the cell. When prompted:  
   - Upload both `KDDTrain.txt` and `KDDTest.txt`

The script will proceed with:  
1. Dataset loading and cleanup  
2. Label encoding for binary classification (`normal` vs. `attack`)  
3. Categorical encoding and numerical feature scaling  
4. Neural network model definition using Keras  
5. Training using early stopping and validation split  
6. Evaluation and metric reporting  
7. Visualization of training progress and classification results 

---

## Output & Interpretation

### Performance Metrics:
- **Classification Report**: Precision, recall, F1-score  
  Provides per-class performance insights — how well the model identifies attacks and normal traffic.
- **Accuracy**: Overall detection performance  
  Percentage of correctly predicted samples.
- **ROC-AUC Score**: Probabilistic measure of classification confidence  
  Indicates how well the model separates classes (1.0 = perfect, 0.5 = random guessing).

### Visualizations:

#### 1. Training Loss & Accuracy Curves  
- Show the model’s learning progress over epochs  
- **Loss curve** should decrease (model improves)  
- **Accuracy curve** should increase  
- Divergence between training and validation curves can indicate overfitting or underfitting

#### 2. Confusion Matrix  
- Displays counts of true positives (correct attacks), true negatives (correct normal), false positives, and false negatives  
- Helps identify if the model is biased toward one class or if specific errors are common

#### 3. ROC Curve  
- Plots True Positive Rate (Recall) vs False Positive Rate  
- Area Under Curve (AUC) reflects overall classification quality  
- Higher AUC indicates better discrimination between normal and attack classes

---

## Model Export

- Model saved as `nsl_dnn_model.h5` (Keras HDF5 format)  
- Can be downloaded from Colab via:

```python
from google.colab import files
files.download("nsl_dnn_model.h5")
```

---

## Key Implementation Notes

- Categorical features (`protocol_type`, `service`, `flag`) are one-hot encoded  
- Numerical features normalized using `StandardScaler`  
- Binary label mapping:  
  - `0 = normal`  
  - `1 = attack`  
- Model architecture:  
  - 3 dense layers with 128, 64, and 32 units  
  - Dropout (0.3) to reduce overfitting  
  - Sigmoid output for binary classification  
- EarlyStopping monitors `val_loss` to avoid overfitting

---

## Contact & Contributions

If adapting this for local execution:  
- Use `tf.keras.utils.get_file()` or direct file paths for dataset loading  
- Replace Colab file upload logic with `pd.read_csv('path/to/KDDTrain.txt')`

For improvements or feedback, open an issue or share a forked version.