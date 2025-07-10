Network Intrusion Detection Using Machine Learning Algorithms
Overview

This project explores the application of Machine Learning (ML) techniques for Network Intrusion Detection, aiming to identify and classify network attacks effectively. The study compares three ML algorithms:

    Random Forest (RF)

    Support Vector Machines (SVM)

    Deep Neural Networks (DNN)

Using the NSL-KDD dataset. Performance metrics such as accuracy, precision, recall, F1 score, and ROC-AUC are used to evaluate each model’s effectiveness.

Project Structure

    Code: Python scripts implementing each algorithm and evaluation metrics.

    Datasets: NSL-KDD dataset used for training and testing the models.

    Reports: Detailed analysis of model performance, including tables, figures, and plots.

Key Features

    Preprocessing of network traffic data to enhance model accuracy.

    Hyperparameter tuning using cross-validation for robust model training.

    Visualization of classification results and feature spaces.

    Comparative performance analysis across RF, SVM, and DNN.

    Detailed discussion on model strengths, weaknesses, and applicability in real-world Intrusion Detection Systems (IDS).

Installation and Usage

    Clone the repository:

git clone https://github.com/zach-mccullough/Network-Intrusion-Detection.git
cd Network-Intrusion-Detection

Create a virtual environment and activate it (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Run the models:

    Each script corresponds to a different ML algorithm.

    Example to run Random Forest:

        python random_forest.py

    View results and plots generated in the output directory.

Dataset

The project uses the NSL-KDD dataset, a widely recognized benchmark for intrusion detection research. The dataset is included in the repository under /datasets.

Performance Summary

| Model           | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------------|----------|-----------|--------|----------|---------|
| Random Forest   | 0.86     | 0.88      | 0.86   | 0.81     | 0.98    |
| SVM             | 0.85     | 0.86      | 0.85   | 0.80     | 0.93    |
| Deep Neural Net | 0.79     | 0.84*     | 0.79*  | 0.79     | 0.93    |

*Weighted averages across classes.


*Weighted averages across classes.
Future Work

    Improve recall on rare attack classes using techniques like SMOTE, focal loss, and class weighting.

    Optimize deep learning architectures for better generalization.

    Explore hybrid models combining machine learning and anomaly detection methods.

References

Please refer to the research paper and documentation included for detailed literature review and citations.
Contact

For questions or collaboration, contact Zachariah McCullough at [zachariahmc96@gmail.com] or open an issue on GitHub.
