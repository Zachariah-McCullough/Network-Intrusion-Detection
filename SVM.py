# ============================================================
# Intrusion Detection using Support Vector Machine (SVM)
# Dataset: NSL-KDD (KDDTrain.txt, KDDTest.txt)
# Purpose: Preprocess data, train SVM, evaluate model performance
# ============================================================

# --- Imports and Setup ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Start runtime tracking
start_time = time.time()

# Scikit-learn modules
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_fscore_support, auc
)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ============================================================
# 1. Label Categorization Helper
# Groups specific KDD attack labels into 5 broad classes
# ============================================================
def categorize_label(label):
    dos = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    probe = ['ipsweep', 'nmap', 'portsweep', 'satan']
    r2l = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack', 'xterm']

    if label == 'normal':
        return 'normal'
    elif label in dos:
        return 'dos'
    elif label in probe:
        return 'probe'
    elif label in r2l:
        return 'r2l'
    elif label in u2r:
        return 'u2r'
    else:
        return 'other'

# ============================================================
# 2. Load and Preprocess Data
# ============================================================

# KDD column names
column_names = [ 
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'label', 'difficulty' 
]

# Set working directory and file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
train_path = os.path.join(script_dir, "KDDTrain.txt")
test_path = os.path.join(script_dir, "KDDTest.txt")

# Load datasets
print(f"Loading train data from: {train_path}")
print(f"Loading test data from: {test_path}")
train_df = pd.read_csv(train_path, header=None, names=column_names)
test_df = pd.read_csv(test_path, header=None, names=column_names)

# Drop irrelevant 'difficulty' column
train_df.drop("difficulty", axis=1, inplace=True)
test_df.drop("difficulty", axis=1, inplace=True)

# Map specific labels to broader categories
train_df["label"] = train_df["label"].apply(categorize_label)
test_df["label"] = test_df["label"].apply(categorize_label)

# ============================================================
# 3. Encode Categorical Features
# protocol_type, service, flag → integers
# ============================================================
cat_cols = ['protocol_type', 'service', 'flag']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    le_dict[col] = le

# Split features (X) and labels (y)
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# ============================================================
# 4. Encode Labels and One-hot Binarize for ROC
# ============================================================
label_le = LabelEncoder()
y_train_enc = label_le.fit_transform(y_train)
y_test_enc = [label_le.transform([y])[0] if y in label_le.classes_ else -1 for y in y_test]

# Remove unknown classes from test
valid_idx = [i for i, y in enumerate(y_test_enc) if y != -1]
X_test = X_test.iloc[valid_idx]
y_test_enc = np.array(y_test_enc)[valid_idx]

# One-hot encode for ROC-AUC multi-class
total_classes = len(label_le.classes_)
y_test_bin = label_binarize(y_test_enc, classes=range(total_classes))

# ============================================================
# 5. Feature Scaling (Standardization)
# Important for SVM
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. Train SVM using GridSearchCV
# with probability outputs enabled
# ============================================================
print("\nTraining SVM with Grid Search...")
svm_params = {
    "C": [1, 10],
    "gamma": ['scale', 0.01],
    "kernel": ['rbf']
}
svm = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1, verbose=1)
svm_grid.fit(X_train_scaled, y_train_enc)
best_svm = svm_grid.best_estimator_
print("Best SVM Parameters:", svm_grid.best_params_)

# ============================================================
# 7. Model Evaluation + ROC Curve & Confusion Matrix
# ============================================================
def evaluate_model(name, model):
    # Predict labels and probabilities
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted')

    # Get probabilities for ROC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test_scaled)
    else:
        y_score = model.decision_function(X_test_scaled)
        if y_score.ndim == 1:
            y_score = np.vstack([1 - y_score, y_score]).T

    # Compute macro-average AUC
    try:
        auc_score = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
    except:
        auc_score = None

    # Print summary
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc_score}")
    print("\nClassification Report:\n", classification_report(y_test_enc, y_pred, target_names=label_le.classes_))

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test_enc, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=label_le.classes_, yticklabels=label_le.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # ROC Curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    for i in range(total_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(total_classes):
        plt.plot(fpr[i], tpr[i], label=f"{label_le.classes_[i]} (AUC = {roc_auc_dict[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score
    }

# Evaluate SVM
svm_results = evaluate_model("SVM", best_svm)

# ============================================================
# 8. Summary Visualization
# ============================================================
results_df = pd.DataFrame({"SVM": svm_results}).T
print("\n🔚 Summary of Results:\n", results_df)

# Quick comparison plot: Accuracy & F1
results_df[["accuracy", "f1"]].plot(kind="bar", figsize=(8, 6), legend=True)
plt.title("Model Comparison: Accuracy & F1 Score")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# 9. Runtime Reporting
# ============================================================
end_time = time.time()
elapsed = end_time - start_time
minutes = elapsed // 60
seconds = elapsed % 60
print(f"\nTotal runtime: {int(minutes)} minutes and {int(seconds)} seconds")