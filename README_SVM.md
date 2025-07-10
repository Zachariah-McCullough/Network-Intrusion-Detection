# SVM Classifier for KDD Intrusion Detection Dataset

This project trains and evaluates a Support Vector Machine (SVM) classifier using the KDD dataset to detect various types of network attacks.

---

## Reproducibility Note

While repeated runs of the code generally produce very similar results, slight variations may occur due to randomness in the training process and differences in system environments. Exact replication of results is therefore not guaranteed. Please allow up to **15 minutes** for runtime.

---

## Requirements

Ensure you have **Python 3.7+** installed. Required packages:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

You can install all required packages with:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Files Needed

Make sure the following files are placed in the same directory as the script (this step should already be complete):

- `KDDTrain.txt`
- `KDDTest.txt`
- `SVM.py`

## How to Run

Open a command line or terminal in the directory containing the files and run:

```bash
python SVM.py
```

## Understanding the Output

Once the script finishes training and evaluating the model, it will display:

1. **Confusion Matrix (Heatmap)**

    - Rows represent the actual class.
    - Columns represent the predicted class.
    - Diagonal cells show correct predictions.
    - Off-diagonal cells indicate misclassifications.
    - For example, a bright square on dos-dos means many DoS attacks were correctly identified. Brightness off-diagonal (e.g., r2l-normal) indicates R2L attacks misclassified as normal.

2. **ROC Curves (One-vs-Rest)**

    - Each curve corresponds to one attack category.
    - Curves closer to the top-left corner indicate better performance.
    - The dashed diagonal line represents random chance.
    - AUC (Area Under Curve) close to 1.0 indicates strong model performance.

3. **Summary Bar Plot**

    - Displays accuracy and F1 score for the model.
    - Useful for quickly gauging overall classification effectiveness.

## Sample Output

```
Best SVM Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

SVM Performance:
Accuracy: 0.8508
Precision: 0.8639
Recall: 0.8508
F1: 0.8010
ROC-AUC: 0.9370
```

## Troubleshooting

- If the script cannot find the dataset files, verify they are located in the same folder as `SVM.py`.
- If you get errors related to missing packages, ensure all dependencies are installed correctly.
- On different systems, runtime may vary depending on hardware.

## Contact

For help, improvements, or questions, feel free to reach out or submit a pull request.

---